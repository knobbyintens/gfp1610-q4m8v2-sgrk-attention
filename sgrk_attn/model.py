"""Linear-attention kernel mixer for the SGRK-Attention paper (2602.05678)."""

import torch
import torch.nn as nn


class AttentionKernelMixer(nn.Module):
    def __init__(self, dim=64, heads=4):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.heads = heads

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scale = q.shape[-1] ** -0.5
        attn = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        return self.proj(attn @ v)
