"""
SegFormer (MLP) decode head: each multi-scale backbone feature is projected to
a common embedding dim with a 1x1 conv, upsampled to the highest-resolution
feature's size (stride/4), concatenated, and fused. Returns both:
  - seg_logits: (B, num_classes, H/4, W/4)  (caller upsamples to full res)
  - fused_feat: (B, decode_dim, H/4, W/4)   the pre-classifier feature, used
                downstream as "Segmentation Features" for cross-attention fusion.

Domain-conditioned via FiLM (same mechanism as the depth branch's
DepthConditioningModule) -- indoor and outdoor share the same 15-class output
space, but the same class can look very different across domains (e.g. a
Cityscapes building wall vs. an indoor room wall are both "wall" but visually
very different). Without any domain signal, one shared classifier has to
resolve that ambiguity purely from texture/context; FiLM gives it an explicit
"which domain am I looking at" signal to condition on, same as depth already
gets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import FiLM


class MLPProject(nn.Module):
    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class SegFormerHead(nn.Module):
    def __init__(self, in_channels_list, embed_dim=256, num_classes=15, dropout=0.1,
                 num_domains=2, film_hidden=128, domain_conditioning=True):
        super().__init__()
        self.projections = nn.ModuleList([MLPProject(c, embed_dim) for c in in_channels_list])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.domain_conditioning = domain_conditioning
        if domain_conditioning:
            self.film = FiLM(num_domains, embed_dim, film_hidden)
        self.dropout = nn.Dropout2d(dropout)
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats, domain_cond=None):
        """
        feats: list of (B, C_i, H_i, W_i), highest-res feature (stride 4) is feats[0]
        domain_cond: (B,) long domain ids OR (B, num_domains) soft probs -- same
            convention as DepthConditioningModule/FiLM. Required if
            domain_conditioning=True (the default); ignored otherwise.
        """
        target_size = feats[0].shape[-2:]
        projected = []
        for feat, proj in zip(feats, self.projections):
            p = proj(feat)
            p = F.interpolate(p, size=target_size, mode="bilinear", align_corners=False)
            projected.append(p)
        fused = self.fuse(torch.cat(projected, dim=1))     # (B, embed_dim, H/4, W/4)
        if self.domain_conditioning:
            assert domain_cond is not None, "domain_cond is required when domain_conditioning=True"
            fused = self.film(fused, domain_cond)
        fused = self.dropout(fused)
        seg_logits = self.classifier(fused)                 # (B, num_classes, H/4, W/4)
        return seg_logits, fused
