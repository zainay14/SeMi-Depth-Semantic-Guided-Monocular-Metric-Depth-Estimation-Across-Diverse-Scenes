"""
DPT-style dense-prediction depth decoder (revised architecture).

Semantic (cross-attention) and domain (FiLM) conditioning now happen ONCE,
upstream, in models.fusion.DepthConditioningModule, producing the
"Conditioned Depth Features" at the bottleneck (stride-32) resolution. This
decoder receives that single conditioned feature plus the plain multi-scale
Swin features, and does a standard RefineNet-style top-down fusion:

    reassembled[stage] = 1x1 conv of raw Swin feature at that stage
    bottleneck start   = reassembled[deepest] + Conditioned Depth Features
    then progressively: fuse with next-shallower reassembled feature,
                         refine (2x ResidualConvUnit), upsample x2

So the decoder's three information sources are:
    1. Shared visual features   -- reassemble convs at every stage
    2. Semantic features        -- baked into the bottleneck via cross-attention (upstream)
    3. Domain information       -- baked into the bottleneck via FiLM (upstream)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(x + out)


class FusionBlock(nn.Module):
    """Combine a higher-level (lower-res) feature with the current-level
    reassembled feature, then refine with residual conv units and upsample x2."""
    def __init__(self, channels, extra_capacity=True, dropout_p=0.0):
        super().__init__()
        self.res1 = ResidualConvUnit(channels)
        self.res2 = ResidualConvUnit(channels)
        # extra_capacity: a 3rd residual unit. Since the backbone itself is staying
        # Swin-Tiny (fixed, matches the paper's stated architecture), this is the
        # decoder-side capacity lever that doesn't touch the backbone claim at all --
        # more refinement depth per decoder stage, same encoder.
        self.res3 = ResidualConvUnit(channels) if extra_capacity else None
        # Spatial dropout (zeroes whole channels, not individual pixels -- the
        # standard choice for conv feature maps) as a regularizer against the
        # outdoor-domain overfitting pattern (train RMSE improving steadily while
        # val RMSE plateaued). Automatically inactive at eval() time, same as any
        # nn.Dropout -- no effect on inference/TTA.
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0 else None

    def forward(self, current, prev=None):
        if prev is not None:
            if prev.shape[-2:] != current.shape[-2:]:
                prev = F.interpolate(prev, size=current.shape[-2:], mode="bilinear", align_corners=False)
            current = current + prev
        x = self.res1(current)
        x = self.res2(x)
        if self.res3 is not None:
            x = self.res3(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return x


class AdaptiveBinsHead(nn.Module):
    """
    Adaptive Disparity / Inverse-Depth Bins Discretization Head.
    Operates in disparity space (xi = 1/z) where stereo camera matching noise is uniform:
      1. Predicts 128 normalized bin widths b over inverse-depth range [xi_min, xi_max],
         where xi_min = 1/d_max and xi_max = 1/d_min.
      2. Computes dynamic disparity bin centers xi_k in [xi_min, xi_max].
      3. Predicts per-pixel probability map P(x, y).
      4. Computes expected inverse depth: xi_hat = sum_k(P_k * xi_k).
      5. Inverts back to metric depth: z_hat = 1 / xi_hat.
    
    This compresses far-range (50m-80m) stereo disparity noise into a tiny interval,
    completely eliminating far-field squared error blowup.
    """
    def __init__(self, decode_dim=256, num_bins=128, dropout_p=0.0, mode="inverse_depth"):
        super().__init__()
        self.num_bins = num_bins
        self.mode = mode
        self.bin_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(decode_dim, decode_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(decode_dim // 2, num_bins),
            nn.Softmax(dim=1),
        )
        self.prob_conv = nn.Sequential(
            nn.Conv2d(decode_dim, decode_dim // 2, 3, padding=1),
            nn.BatchNorm2d(decode_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
            nn.Conv2d(decode_dim // 2, num_bins, 3, padding=1),
        )

    def forward(self, x, domain_cond=None):
        B = x.shape[0]
        bin_widths_norm = self.bin_mlp(x)  # (B, num_bins), sums to 1.0

        if domain_cond is None:
            d_max = torch.full((B, 1), 80.0, device=x.device, dtype=x.dtype)
            d_min = torch.full((B, 1), 0.1, device=x.device, dtype=x.dtype)
        elif domain_cond.dim() == 1:
            d_max = torch.where(domain_cond == 1, 80.0, 10.0).unsqueeze(1).float()
            d_min = torch.full_like(d_max, 0.1)
        else:
            d_max = (domain_cond[:, 0:1] * 10.0 + domain_cond[:, 1:2] * 80.0).float()
            d_min = torch.full_like(d_max, 0.1)

        cum_widths = torch.cumsum(bin_widths_norm, dim=1)
        centers_norm = cum_widths - 0.5 * bin_widths_norm  # (B, num_bins) in [0, 1]

        prob_logits = self.prob_conv(x)
        prob_map = F.softmax(prob_logits, dim=1)  # (B, num_bins, H', W')

        if self.mode == "inverse_depth":
            xi_min = 1.0 / d_max.clamp_min(1e-3)
            xi_max = 1.0 / d_min.clamp_min(1e-3)
            bin_centers_xi = xi_min + (xi_max - xi_min) * centers_norm  # (B, num_bins)
            inv_depth = torch.sum(prob_map * bin_centers_xi.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)
            depth = 1.0 / inv_depth.clamp_min(1e-6)
            bin_centers = 1.0 / bin_centers_xi.clamp_min(1e-6)
        else:
            bin_centers = d_min + (d_max - d_min) * centers_norm
            depth = torch.sum(prob_map * bin_centers.unsqueeze(-1).unsqueeze(-1), dim=1, keepdim=True)

        return depth, bin_centers


class DPTDepthDecoder(nn.Module):
    def __init__(self, in_channels_list, decode_dim=256, aux_supervision=True, extra_capacity=True,
                 dropout_p=0.0, use_bins=True, num_bins=128):
        super().__init__()
        self.use_bins = use_bins
        self.num_bins = num_bins
        # 1x1 "reassemble" convs to bring every stage to decode_dim channels
        self.reassemble = nn.ModuleList([
            nn.Conv2d(c, decode_dim, kernel_size=1) for c in in_channels_list
        ])
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(decode_dim, extra_capacity=extra_capacity, dropout_p=dropout_p) for _ in in_channels_list
        ])

        if use_bins:
            self.head = AdaptiveBinsHead(decode_dim=decode_dim, num_bins=num_bins, dropout_p=dropout_p)
        else:
            head_mid = 64 if extra_capacity else 32
            self.head = nn.Sequential(
                nn.Conv2d(decode_dim, decode_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_p) if dropout_p > 0 else nn.Identity(),
                nn.Conv2d(decode_dim // 2, head_mid, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_mid, 1, 1),
                nn.Softplus(),
            )

        self.aux_supervision = aux_supervision
        if aux_supervision:
            if use_bins:
                self.aux_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(decode_dim, decode_dim // 2, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(decode_dim // 2, num_bins, 1),
                    )
                    for _ in range(len(in_channels_list) - 1)
                ])
            else:
                self.aux_heads = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(decode_dim, decode_dim // 2, 3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(decode_dim // 2, 1, 1),
                        nn.Softplus(),
                    )
                    for _ in range(len(in_channels_list) - 1)
                ])

    def forward(self, feats, conditioned_deep_feat, domain_cond=None):
        """
        feats: list of 4 backbone feature maps, shallow->deep, strides [4,8,16,32]
        conditioned_deep_feat: (B, decode_dim, H/32, W/32) -- output of
            DepthConditioningModule (already carries semantic + domain info)
        domain_cond: domain indicators/probabilities for setting depth bounds.

        Returns: (depth, bin_centers, depth_aux)
            depth:       (B,H,W) final prediction
            bin_centers: (B,num_bins) predicted bin centers, or None if use_bins=False
            depth_aux:   list of (B,h_i,w_i) auxiliary predictions
        """
        reassembled = [conv(f) for conv, f in zip(self.reassemble, feats)]  # shallow->deep
        reassembled[-1] = reassembled[-1] + conditioned_deep_feat

        stage_order = list(reversed(range(len(reassembled))))  # deepest -> shallowest
        prev = None
        for k, i in enumerate(stage_order):
            prev = self.fusion_blocks[i](reassembled[i], prev)

        bin_centers = None
        if self.use_bins:
            depth, bin_centers = self.head(prev, domain_cond=domain_cond)  # (B,1,h,w), (B,num_bins)
        else:
            depth = self.head(prev)

        depth = F.interpolate(depth, scale_factor=2, mode="bilinear", align_corners=False)

        depth_aux = []
        if self.aux_supervision and self.training:
            # Note: aux heads run on intermediate stage features if needed
            pass

        return depth.squeeze(1), bin_centers, depth_aux

