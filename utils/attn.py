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


class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) block used in transformer architectures.

    Args:
        dim (int): Input and output dimension.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 4 * dim.
        drop (float, optional): Dropout probability. Defaults to 0.0.

    Forward Input:
        x (Tensor): Shape (B, N, C)

    Forward Output:
        Tensor: Shape (B, N, C)
    """
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """
    Local self-attention applied within non-overlapping windows.

    Args:
        dim (int): Input dimension.
        window_size (int): Size of the attention window (assumed square).
        num_heads (int): Number of attention heads.

    Forward Input:
        x (Tensor): Shape (B, N, C), where N is a perfect square (H*W).

    Forward Output:
        Tensor: Shape (B, N, C)
    """
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "Input tokens do not form a square feature map."

        x = x.view(B, H, W, C)

        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "H and W must be divisible by window_size"

        x = x.reshape(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)

        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        out = self.proj(out)

        out = out.view(B, H // ws, W // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)
        out = out.reshape(B, N, C)

        return out


class GridAttention(nn.Module):
    """
    Global self-attention across spatial grid tokens.
    Operates on coarse grid-level summary tokens extracted from the feature map.

    Args:
        dim (int): Input dimension.
        grid_size (int): Grid size for spatial partitioning.
        num_heads (int): Number of attention heads.

    Forward Input:
        x (Tensor): Shape (B, N, C), where N is a perfect square (H*W).

    Forward Output:
        Tensor: Shape (B, N, C)
    """
    def __init__(self, dim, grid_size=7, num_heads=8):
        super().__init__()
        self.grid_size = grid_size
        self.num_heads = num_heads
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(0.0)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "Input tokens do not form a square feature map."

        x = x.view(B, H, W, C)

        G = self.grid_size
        assert H % G == 0 and W % G == 0, "H and W must be divisible by grid_size"

        x = x.reshape(B, G, H // G, G, W // G, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, G * G, (H // G) * (W // G), C)

        grid_tokens = x.mean(dim=2)

        qkv = self.qkv(grid_tokens).reshape(B, G * G, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, G * G, C)
        out = self.proj(out)

        out = out.unsqueeze(2).expand(-1, -1, (H // G) * (W // G), -1)
        out = out.view(B, G, G, H // G, W // G, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, H, W, C).reshape(B, N, C)

        return out


class MaxViTBlock(nn.Module):
    """
    MaxViT block designed for flattened CLIP patch tokens.

    Combines local Window Attention and global Grid Attention (checkerboard-style),
    each with residual connections. Followed by a LayerNorm + MLP block. 
    No MBConv is used. Both windows and grids default to size 7 (P=G=7).

    Reference:
        Tu et al., "MaxViT: Multi-Axis Vision Transformer", ECCV 2022.
        
    Architecture:
        x -> LN -> WindowAttention -> Add
          -> LN -> GridAttention -> Add
          -> LN -> MLP -> Add

    Args:
        dim (int): Input and output dimension.
        window_size (int): Window size for WindowAttention.
        grid_size (int): Grid size for GridAttention.
        num_heads (int): Number of attention heads for both attentions.
        mlp_ratio (float): Expansion ratio for MLP hidden layer. Defaults to 4.0.
        drop (float): Dropout rate used in MLP and attention.

    Forward Input:
        x (Tensor): Shape (B, N, C), where N is H * W and must be square.

    Forward Output:
        Tensor: Shape (B, N, C)
    """
    def __init__(self, dim, window_size=7, grid_size=7, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.window_attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.grid_attn = GridAttention(dim, grid_size, num_heads)

        self.norm3 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x):
        # Window Attention block
        x = x + self.window_attn(self.norm1(x))

        # Grid Attention block
        x = x + self.grid_attn(self.norm2(x))

        # MLP block
        x = x + self.mlp(self.norm3(x))

        return x
    

if __name__ == "__main__":
    B, C = 4, 768  # Batch size and channel dimension

    for size in [7, 14]:
        N = size * size # Number of patches
        x = torch.randn(B, N, C)
        print(f"\nTesting MaxViTBlock with input size: {size}x{size} (N={N})")

        # Test MaxViTBlock
        print("MaxViTBlock:")
        block = MaxViTBlock(dim=C, window_size=7, grid_size=7, num_heads=8)
        out_maxvit = block(x)

        print(f"- Input shape: {x.shape}")
        # print(f"- Output Tensor (MaxViTBlock): {out_maxvit}")
        print(f"- Output shape: {out_maxvit.shape}")
        assert out_maxvit.shape == x.shape, "Output shape mismatch!"
        print("- Passed shape check.")

        # Test CBAM
        print("CBAM:")
        cbam = CBAM(dim=C, ratio=8)
        out_cbam = cbam(x)
        print(f"- Input shape:  {x.shape}")
        # print(f"- Output Tensor (CBAM): {out_cbam}")
        print(f"- Output shape: {out_cbam.shape}")
        assert out_cbam.shape == x.shape, "CBAM output shape mismatch!"
        print("- Passed shape check.")