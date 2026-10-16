"""
MultiTaskModel: wires the complete pipeline from the diagram end-to-end.

    Input Video/Image
        -> Swin Transformer Tiny (Shared Encoder)
        -> Domain Head            (indoor/outdoor logits, from deepest feature)
        -> SegFormer Head         (segmentation logits + fused "Segmentation Features")
        -> Depth Features (deepest Swin feature, projected)
              -> Cross-Attention with Segmentation Features -> Semantic-aware Depth Features
              -> FiLM with Domain Embedding                 -> Conditioned Depth Features
        -> DPT Decoder (fuses Conditioned Depth Features + all-stage shared visual
                         features -> Predicted Depth Map)
        -> Boundary Consistency Loss (computed outside the model, in utils/losses.py,
                                       using seg_logits + predicted depth)

    This matches the revised architecture: conditioning (cross-attention +
    FiLM) happens ONCE at the bottleneck, before the DPT decoder, rather than
    being re-applied at every decoder stage.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import SwinTinyBackbone
from models.domain_head import DomainHead
from models.seg_head import SegFormerHead
from models.fusion import DepthConditioningModule
from models.depth_decoder import DPTDepthDecoder


class MultiTaskModel(nn.Module):
    def __init__(self, model_cfg, num_seg_classes, img_size=512):
        super().__init__()
        self.backbone = SwinTinyBackbone(model_cfg.backbone_name, model_cfg.pretrained, img_size=img_size)
        ch = self.backbone.channels

        self.domain_head = DomainHead(ch[-1], model_cfg.domain_hidden, model_cfg.num_domains)
        self.seg_head = SegFormerHead(ch, model_cfg.decode_dim, num_seg_classes,
                                       num_domains=model_cfg.num_domains, film_hidden=model_cfg.film_hidden,
                                       domain_conditioning=model_cfg.seg_domain_conditioning)
        self.depth_conditioning = DepthConditioningModule(
            ch[-1], model_cfg.decode_dim, model_cfg.num_domains,
            model_cfg.film_hidden, model_cfg.cross_attn_heads,
        )
        self.depth_decoder = DPTDepthDecoder(ch, model_cfg.decode_dim,
                                              aux_supervision=model_cfg.depth_aux_supervision,
                                              extra_capacity=model_cfg.depth_decoder_extra_capacity,
                                              dropout_p=model_cfg.depth_decoder_dropout,
                                              use_bins=getattr(model_cfg, "use_depth_bins", True),
                                              num_bins=getattr(model_cfg, "num_depth_bins", 128))
        self.num_seg_classes = num_seg_classes

    def forward(self, x, domain_gt=None, teacher_force_domain=True, return_features=False):
        """
        x: (B,3,H,W) normalized image
        domain_gt: (B,) long ground-truth domain, used for FiLM conditioning during
                   training (teacher forcing). At inference, the model's own
                   predicted domain probabilities are used instead.
        return_features: if True, also returns pooled bottleneck ("domain_feat") and
                   pooled segmentation ("seg_feat_pooled") embeddings for downstream
                   analysis (e.g. visualize_tsne.py). Adds negligible cost, no effect
                   on training/loss.
        """
        H, W = x.shape[-2:]
        feats = self.backbone(x)                       # 4 feature maps, strides [4,8,16,32]

        domain_logits, domain_feat = self.domain_head(feats[-1], return_features=True)  # (B,num_domains), (B,C)

        # Compute domain_cond BEFORE the seg head now (previously computed after, since
        # only depth used it) -- both seg and depth need it for their respective FiLM
        # conditioning, and it only depends on domain_logits/domain_gt, not on seg_head's
        # output, so there's no ordering issue moving it up.
        if self.training and teacher_force_domain and domain_gt is not None:
            domain_cond = domain_gt
        else:
            domain_cond = F.softmax(domain_logits, dim=1)   # soft mixture at inference

        seg_logits_low, seg_feat = self.seg_head(feats, domain_cond=domain_cond)   # (B,C,H/4,W/4), (B,decode_dim,H/4,W/4)
        seg_logits = F.interpolate(seg_logits_low, size=(H, W), mode="bilinear", align_corners=False)

        # Depth Features -> Cross-Attention (w/ Segmentation Features) -> Semantic-aware
        # Depth Features -> FiLM (w/ domain embedding) -> Conditioned Depth Features
        conditioned_deep_feat = self.depth_conditioning(feats[-1], seg_feat, domain_cond)
        depth, bin_centers, depth_aux = self.depth_decoder(feats, conditioned_deep_feat, domain_cond=domain_cond)
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)

        out = {
            "domain_logits": domain_logits,
            "seg_logits": seg_logits,
            "depth": depth,
            "bin_centers": bin_centers,
            "depth_aux": depth_aux,   # list of coarser-resolution aux predictions, see utils/losses.py
        }
        if return_features:
            out["domain_feat"] = domain_feat                                     # (B, C_backbone_deep)
            out["seg_feat_pooled"] = F.adaptive_avg_pool2d(seg_feat, 1).flatten(1)  # (B, decode_dim)
        return out

    def get_param_groups(self, base_lr, backbone_lr_mult, weight_decay, protect_seg=False):
        """Separate param groups so the pretrained backbone and segmentation heads
        fine-tune slower than new depth heads."""
        backbone_params, seg_params, head_params = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            elif protect_seg and (name.startswith("seg_head.") or name.startswith("domain_head.")):
                seg_params.append(p)
            else:
                head_params.append(p)
        groups = [
            {"params": backbone_params, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay},
            {"params": head_params, "lr": base_lr, "weight_decay": weight_decay},
        ]
        if seg_params:
            groups.append({"params": seg_params, "lr": base_lr * backbone_lr_mult, "weight_decay": weight_decay})
        return groups
