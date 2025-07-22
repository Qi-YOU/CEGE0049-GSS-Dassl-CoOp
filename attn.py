"""
Attention modules for CLIP-based adaptation, including:
- **CBAM** (Convolutional Block Attention Module) [1]:
    Simple attention module of channel/spatial attention for CNNs.
- **Multi-Axis Attention** [2]: #TODO: Implementation
    Hybrid local/global attention for vision transformers.  

This module implements attention mechanisms to enhance spatial and channel-wise feature
adaptation in CLIP-based models. Designed for compatibility with CLIP's vision encoders.

References:
[1] Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).
[2] Tu, Z., Talebi, H., Zhang, H., Yang, F., Milanfar, P., Bovik, A., & Li, Y. (2022, October). Maxvit: Multi-axis vision transformer. In European conference on computer vision (pp. 459-479). Cham: Springer Nature Switzerland.

"""


import torch
import torch.nn as nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention module with avg and max pooling, followed by MLP.
    """
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B, N, C]
        x_perm = x.permute(0, 2, 1)  # [B, C, N]
        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))
        out = avg_out + max_out
        scale = self.sigmoid(out).unsqueeze(1)  # [B, 1, C]
        return x * scale  # [B, N, C]


class SpatialAttention(nn.Module):
    """
    Spatial attention module using channel-wise avg/max and 1D convolution.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [B, N, C]
        x_perm = x.permute(0, 2, 1)  # [B, C, N]
        avg_out = torch.mean(x_perm, dim=1, keepdim=True)
        max_out, _ = torch.max(x_perm, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, N]
        attn = self.sigmoid(self.conv(x_cat))  # [B, 1, N]
        return x * attn.permute(0, 2, 1)  # [B, N, C]


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) combining channel and spatial attention.
    References: Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19).
    """
    def __init__(self, dim, ratio=8):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(dim, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
