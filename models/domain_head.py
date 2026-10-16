"""Linear domain classification head: indoor(0) / outdoor(1)."""
import torch
import torch.nn as nn


class DomainHead(nn.Module):
    def __init__(self, in_channels, hidden=128, num_domains=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_domains),
        )

    def forward(self, deepest_feat, return_features=False):
        # deepest_feat: (B, C, H, W) -- last backbone stage
        x = self.pool(deepest_feat).flatten(1)   # (B, C), pooled bottleneck feature
        logits = self.mlp(x)                      # (B, num_domains)
        if return_features:
            return logits, x
        return logits
