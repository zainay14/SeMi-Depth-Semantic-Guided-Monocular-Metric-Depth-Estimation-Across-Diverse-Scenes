"""Shared Swin Transformer Tiny encoder producing 4 multi-scale feature maps."""
import timm
import torch
import torch.nn as nn


class SwinTinyBackbone(nn.Module):
    def __init__(self, name="swin_tiny_patch4_window7_224", pretrained=True, img_size=512):
        super().__init__()
        # features_only=True -> returns a list of feature maps at strides [4,8,16,32]
        #
        # BUG FIX: timm's swin_tiny_patch4_window7_224 ships with fixed_input_size=True
        # (baked in from its 224x224 ImageNet pretraining config) and asserts the input
        # matches that unless you explicitly tell it otherwise at construction time. Since
        # this pipeline trains at data_cfg.img_size=512 (not 224), the backbone must be
        # built with img_size=512 (which timm uses to rebuild the absolute position
        # embeddings for that resolution) -- otherwise the very first forward pass raises
        # "Input height (512) doesn't match model (224)."
        self.backbone = timm.create_model(
            name, pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3), img_size=img_size,
        )
        self.channels = self.backbone.feature_info.channels()   # e.g. [96,192,384,768]
        self.strides = self.backbone.feature_info.reduction()    # e.g. [4,8,16,32]

    def forward(self, x):
        feats = self.backbone(x)
        # timm's swin (NHWC) features_only outputs are channel-last (B,H,W,C) for
        # some versions; normalize to channel-first (B,C,H,W) for the rest of the pipeline.
        out = []
        for f in feats:
            if f.dim() == 4 and f.shape[-1] in self.channels:
                f = f.permute(0, 3, 1, 2).contiguous()
            out.append(f)
        return out  # list of 4 tensors, (B, C_i, H/s_i, W/s_i)
