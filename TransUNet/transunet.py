"""
TransUNet for Single-Shot Fringe Projection Profilometry
Input: 960x960 grayscale fringe pattern
Output: 960x960 depth map

Identical to UNet except the bottleneck convolutional block is replaced
with a Vision Transformer (ViT). The encoder and decoder are the same
DoubleConv/DownSample/UpSample blocks as UNet. At the bottleneck
(60x60 spatial, 512 channels in), the feature map is projected to the
transformer embedding dimension, split into patches, processed by
multi-head self-attention + MLP blocks, then reshaped back to spatial
dimensions and projected to 1024 channels before the decoder.
"""

import torch
import torch.nn as nn
import math


# =============================================================================
# Shared UNet building blocks (identical to unet.py)
# =============================================================================

class DoubleConv(nn.Module):
    """Two 3x3 convolutions with InstanceNorm + ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """Downsampling block: DoubleConv + MaxPool."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x


class UpSample(nn.Module):
    """Upsampling block: ConvTranspose + Concat + DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# =============================================================================
# Vision Transformer bottleneck
# =============================================================================

class PatchEmbedding(nn.Module):
    """Flatten spatial feature map into patch tokens."""
    def __init__(self, in_channels, embed_dim, patch_size, num_patches):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.proj(x)                          # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)          # (B, num_patches, embed_dim)
        x = x + self.pos_embed
        return x


class TransformerBlock(nn.Module):
    """Standard ViT transformer block: LayerNorm + MHSA + LayerNorm + MLP."""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBottleneck(nn.Module):
    """
    Vision Transformer bottleneck for TransUNet.

    Takes the encoder output (B, 512, 60, 60), projects to embed_dim,
    splits into patches, runs through transformer layers, then reshapes
    back to spatial dimensions and projects to 1024 channels.

    Patch size 6 at 60x60 gives 100 tokens (10x10 grid).
    """
    def __init__(
        self,
        in_channels=512,
        out_channels=1024,
        spatial_size=60,
        patch_size=6,
        embed_dim=768,
        num_layers=8,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.patch_size = patch_size
        self.grid_size = spatial_size // patch_size  # 10
        self.num_patches = self.grid_size ** 2       # 100

        # Patch embedding: (B, 512, 60, 60) -> (B, 100, 768)
        self.patch_embed = PatchEmbedding(
            in_channels, embed_dim, patch_size, self.num_patches
        )

        # Transformer layers
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Project back: (B, 100, 768) -> (B, out_channels, 60, 60)
        self.proj_out = nn.Sequential(
            nn.Linear(embed_dim, out_channels * patch_size * patch_size),
        )
        self.out_channels = out_channels

    def forward(self, x):
        B = x.shape[0]

        # Embed patches
        x = self.patch_embed(x)       # (B, 100, 768)

        # Transformer
        x = self.blocks(x)            # (B, 100, 768)
        x = self.norm(x)

        # Reshape back to spatial
        x = self.proj_out(x)          # (B, 100, 1024*6*6)
        x = x.reshape(
            B, self.grid_size, self.grid_size,
            self.out_channels, self.patch_size, self.patch_size
        )
        # (B, grid, grid, C, p, p) -> (B, C, grid*p, grid*p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(
            B, self.out_channels, self.spatial_size, self.spatial_size
        )

        return x


# =============================================================================
# TransUNet model
# =============================================================================

class TransUNetFPP(nn.Module):
    """
    TransUNet for Fringe Projection Profilometry.
    Identical to UNetFPP except the bottleneck DoubleConv is replaced
    with a ViT transformer.

    Input:  (B, 1, 960, 960)
    Output: (B, 1, 960, 960)
    """

    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.5):
        super().__init__()

        # ENCODER (identical to UNet)
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # BOTTLENECK (ViT instead of DoubleConv)
        self.bottleneck = ViTBottleneck(
            in_channels=512,
            out_channels=1024,
            spatial_size=60,
            patch_size=6,       # 60/6 = 10x10 grid = 100 tokens
            embed_dim=768,
            num_layers=8,
            num_heads=12,
            mlp_ratio=4.0,
            dropout=0.1,
        )
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER (identical to UNet)
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # OUTPUT
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for conv layers, default init for transformer."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.out(x)


# =============================================================================
# Loss functions (identical to unet.py)
# =============================================================================

class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss (includes background)"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)


class MaskedRMSELoss(nn.Module):
    """RMSE loss that ignores background pixels (depth=0)"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        mask = (target > 0).float()
        squared_error = (pred - target) ** 2
        masked_squared_error = squared_error * mask
        num_valid = mask.sum().clamp(min=self.eps)
        mse = masked_squared_error.sum() / num_valid
        return torch.sqrt(mse + self.eps)


class HybridRMSELoss(nn.Module):
    """Masked RMSE + weak global RMSE anchor."""
    def __init__(self, alpha=0.9, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        mask = (target > 0).float()
        diff = (pred - target) * mask
        mse_masked = (diff * diff).sum() / mask.sum().clamp(min=self.eps)
        masked_rmse = torch.sqrt(mse_masked + self.eps)
        mse_global = ((pred - target) ** 2).mean()
        global_rmse = torch.sqrt(mse_global + self.eps)
        return self.alpha * masked_rmse + (1 - self.alpha) * global_rmse


class L1Loss(nn.Module):
    """Simple L1 loss (Mean Absolute Error)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.abs(pred - target).mean()


class MaskedL1Loss(nn.Module):
    """Masked L1 loss that only considers valid pixels (where target > 0)."""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        mask = (target > 0).float()
        diff = torch.abs(pred - target) * mask
        num_valid = mask.sum().clamp(min=self.eps)
        return diff.sum() / num_valid


class HybridL1Loss(nn.Module):
    """Masked L1 + weak global L1 anchor."""
    def __init__(self, alpha=0.9, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, pred, target):
        mask = (target > 0).float()
        diff = torch.abs(pred - target) * mask
        masked_l1 = diff.sum() / mask.sum().clamp(min=self.eps)
        global_l1 = torch.abs(pred - target).mean()
        return self.alpha * masked_l1 + (1 - self.alpha) * global_l1
