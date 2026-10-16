"""
CrossAttentionFusion: lets segmentation features guide depth features.
  query = depth features, key/value = segmentation features (both flattened to
  tokens over their spatial grid). Standard multi-head scaled dot-product
  attention, residual-added back onto the depth stream.

FiLM: Feature-wise Linear Modulation. A learned embedding for the
  {indoor, outdoor} domain (either the ground-truth label during training with
  teacher forcing, or the domain-head's soft prediction) is mapped to a
  per-channel (gamma, beta) pair that rescales/shifts the depth-decoder
  features: out = feat * (1 + gamma) + beta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.gamma = nn.Parameter(torch.zeros(1))   # learnable residual gate, starts at 0

    def forward(self, depth_feat, seg_feat):
        """
        depth_feat: (B, C, H, W)  -- query stream
        seg_feat:   (B, C, H', W') -- key/value stream (segmentation features)
        returns:    (B, C, H, W) depth features refined by segmentation context
        """
        B, C, H, W = depth_feat.shape
        if seg_feat.shape[-2:] != (H, W):
            seg_feat = F.interpolate(seg_feat, size=(H, W), mode="bilinear", align_corners=False)

        q = depth_feat.flatten(2).transpose(1, 2)     # (B, HW, C)
        kv = seg_feat.flatten(2).transpose(1, 2)        # (B, HW, C)

        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        attn_out, attn_weights = self.attn(q_n, kv_n, kv_n, need_weights=False)
        attn_out = self.out_proj(attn_out)

        fused = q + self.gamma * attn_out               # residual, gated
        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        return fused


class FiLM(nn.Module):
    def __init__(self, num_domains, feat_channels, hidden=128):
        super().__init__()
        self.embed = nn.Embedding(num_domains, hidden)
        self.to_scale_shift = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, feat_channels * 2),
        )

    def forward(self, feat, domain_idx_or_probs):
        """
        feat: (B, C, H, W)
        domain_idx_or_probs: either (B,) long domain ids, or (B, num_domains) soft probs
        """
        if domain_idx_or_probs.dim() == 1:
            emb = self.embed(domain_idx_or_probs)                      # (B, hidden)
        else:
            emb = domain_idx_or_probs @ self.embed.weight               # soft mixture (B, hidden)
        gamma_beta = self.to_scale_shift(emb)                            # (B, 2C)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta


class DepthConditioningModule(nn.Module):
    """
    Pre-decoder conditioning block (revised architecture):

        Depth Features (deepest Swin feature, projected)
            -> Cross-Attention with Segmentation Features -> Semantic-aware Depth Features
            -> FiLM with Domain Embedding                 -> Conditioned Depth Features

    Runs ONCE at the bottleneck, before the DPT decoder, instead of being
    re-applied at every decoder fusion stage. The DPT decoder then receives:
      1. shared multi-scale visual features straight from the Swin encoder
         (via its own reassemble convs), and
      2. this single semantic+domain-conditioned bottleneck feature, which it
         fuses in at the deepest stage and propagates upward.
    """
    def __init__(self, in_channels, decode_dim, num_domains, film_hidden=128, cross_attn_heads=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, decode_dim, kernel_size=1)  # "Depth Features"
        self.cross_attn = CrossAttentionFusion(decode_dim, num_heads=cross_attn_heads)
        self.film = FiLM(num_domains, decode_dim, film_hidden)

    def forward(self, deepest_feat, seg_feat, domain_idx_or_probs):
        depth_feat = self.proj(deepest_feat)                              # Depth Features
        semantic_aware = self.cross_attn(depth_feat, seg_feat)             # Semantic-aware Depth Features
        conditioned = self.film(semantic_aware, domain_idx_or_probs)       # Conditioned Depth Features
        return conditioned
