import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    A basic block: (Conv2d → ReLU → [BatchNorm]) × 2.
    """
    def __init__(self, in_channels, out_channels, batchnorm: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """
    Downscaling: MaxPool2d(stride=2) → ConvBlock.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    """
    Upscaling: ConvTranspose2d(stride=2) to double spatial dims, 
    then concatenate skip features, then ConvBlock.
    """
    def __init__(self, in_channels, out_channels, batchnorm: bool = True):
        """
        in_channels: number of channels coming from previous layer
        out_channels: number of channels in the skip connection (and also the desired output of this block)
        """
        super().__init__()
        # First, upsample “in_channels → out_channels” by factor 2
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, the input to ConvBlock has (out_channels + out_channels) = 2*out_channels
        self.conv = ConvBlock(in_channels=out_channels * 2, out_channels=out_channels, batchnorm=batchnorm)

    def forward(self, x_from_below, x_skip):
        # x_from_below: (N, in_channels, H, W) → upsampled to (N, out_channels, 2H, 2W)
        x1 = self.up(x_from_below)
        # x_skip: (N, out_channels, 2H, 2W), typically exactly the same spatial dims after proper pooling
        # But if sizes differ by 1 pixel (odd/even), pad x1 so it matches x_skip:
        diffY = x_skip.size(2) - x1.size(2)
        diffX = x_skip.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2,
            ])
        # Now concatenate along channel‐dim: (N, out + out, H, W)
        x = torch.cat([x_skip, x1], dim=1)
        return self.conv(x)


class UNet2D(nn.Module):
    """
    A 2D U-Net where time, color, and z are treated as part of the input channel dimension.
    
    This implementation uses only 2D convolutions. It consists of an encoder that reduces
    spatial dimensions while increasing feature channels, a bottleneck, and a decoder that
    upsamples and merges features with skip connections from the encoder.
    
    Parameters:
        in_channels: number of input channels (e.g. time × color × z)
        out_channels: number of output channels (e.g. same as in_channels for reconstruction)
        base_channels: number of channels after the first conv block
        depth: number of downsampling/upsampling levels
        batchnorm: whether to use BatchNorm2d after each conv
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 2,
        batchnorm: bool = True,
    ):
        super().__init__()

        # Initial convolution to lift input into feature space
        self.in_conv = ConvBlock(in_channels, base_channels, batchnorm=batchnorm)

        # Build the encoder: each Down halves spatial size and doubles channel count
        self.downs = nn.ModuleList()
        ch = base_channels
        self.encoder_channels = [ch]
        for _ in range(depth):
            next_ch = ch * 2
            self.downs.append(Down(ch, next_ch))
            self.encoder_channels.append(next_ch)
            ch = next_ch

        # Bottleneck with double the last encoder channel size
        self.bottleneck = ConvBlock(ch, ch * 2, batchnorm=batchnorm)
        ch = ch * 2

        # Build the decoder: upsample and merge with skip connections
        self.ups = nn.ModuleList()
        for skip_ch in reversed(self.encoder_channels):
            self.ups.append(Up(ch, skip_ch, batchnorm=batchnorm))
            ch = skip_ch

        # Final projection to the output channel space
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect encoder outputs for skip connections
        skips = []
        x = self.in_conv(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottom of the U
        x = self.bottleneck(x)

        # Reverse skip list to align with decoder order
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlockGN(nn.Module):
    """
    A basic block replacing BatchNorm with GroupNorm:
    (Conv2d → ReLU → [GroupNorm]) × 2.
    """
    def __init__(self, in_channels, out_channels, use_gn: bool = True, gn_channels_per_group: int = 8):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_gn:
            num_groups = max(1, out_channels // gn_channels_per_group)
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        if use_gn:
            num_groups = max(1, out_channels // gn_channels_per_group)
            layers.append(nn.GroupNorm(num_groups=num_groups, num_channels=out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DownGN(nn.Module):
    """
    Downscaling: MaxPool2d(stride=2) → ConvBlockGN.
    """
    def __init__(self, in_channels, out_channels, use_gn: bool = True, gn_channels_per_group: int = 8):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlockGN(in_channels, out_channels, use_gn=use_gn, gn_channels_per_group=gn_channels_per_group)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class UpGN(nn.Module):
    """
    Upscaling: ConvTranspose2d(stride=2) to double spatial dims,
    then concatenate skip features, then ConvBlockGN.
    """
    def __init__(self, in_channels, out_channels, use_gn: bool = True, gn_channels_per_group: int = 8):
        super().__init__()
        # First, upsample “in_channels → out_channels” by factor 2
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation, the input to ConvBlockGN has (out_channels + out_channels) = 2*out_channels
        self.conv = ConvBlockGN(in_channels=out_channels * 2,
                                out_channels=out_channels,
                                use_gn=use_gn,
                                gn_channels_per_group=gn_channels_per_group)

    def forward(self, x_from_below, x_skip):
        x1 = self.up(x_from_below)
        diffY = x_skip.size(2) - x1.size(2)
        diffX = x_skip.size(3) - x1.size(3)
        if diffY != 0 or diffX != 0:
            x1 = F.pad(x1, [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2,
            ])
        x = torch.cat([x_skip, x1], dim=1)
        return self.conv(x)


class UNet2D_GN(nn.Module):
    """
    A 2D U-Net where batch normalization is replaced by GroupNorm.
    Time, color, and z are flattened into channels as before.

    Parameters:
        in_channels: number of input channels (time × color × z).
        out_channels: number of output channels.
        base_channels: number of channels after the first ConvBlockGN.
        depth: number of down/up-sampling levels.
        use_gn: whether to use GroupNorm (default True).
        gn_channels_per_group: approx. how many channels per group in GroupNorm.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 2,
        use_gn: bool = True,
        gn_channels_per_group: int = 8,
    ):
        super().__init__()

        # Initial convolution to lift input into feature space
        self.in_conv = ConvBlockGN(
            in_channels,
            base_channels,
            use_gn,
            gn_channels_per_group
        )

        # Build the encoder: each DownGN halves spatial size and doubles channel count
        self.downs = nn.ModuleList()
        ch = base_channels
        self.encoder_channels = [ch]
        for _ in range(depth):
            next_ch = ch * 2
            self.downs.append(DownGN(ch, next_ch, use_gn, gn_channels_per_group))
            self.encoder_channels.append(next_ch)
            ch = next_ch

        # Bottleneck with double the last encoder channel size
        self.bottleneck = ConvBlockGN(
            ch,
            ch * 2,
            use_gn,
            gn_channels_per_group
        )
        ch = ch * 2

        # Build the decoder: upsample and merge with skip connections
        self.ups = nn.ModuleList()
        for skip_ch in reversed(self.encoder_channels):
            self.ups.append(UpGN(ch, skip_ch, use_gn, gn_channels_per_group))
            ch = skip_ch

        # Final projection to the output channel space
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect encoder outputs for skip connections
        skips = []
        x = self.in_conv(x)
        skips.append(x)

        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottom of the U
        x = self.bottleneck(x)

        # Reverse skip list to align with decoder order
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)

        return self.out_conv(x)
