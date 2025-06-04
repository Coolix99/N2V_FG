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
    A shallow 2D U‐Net that treats time, color, z collectively as “channels.”
    - in_channels: (#time × #color × #z)
    - out_channels: however many feature maps or segmentation classes you need
    - base_channels: number of channels after the first ConvBlock
    - depth: number of pooling steps (will create depth+1 encoder blocks and depth+1 decoder blocks)
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

        # 1) Build the “encoder” list of ConvBlocks + Down layers.
        #    We keep track of each stage’s output channels in `enc_channels`.
        enc_channels = []
        # Initial “conv” (no pooling) from in_channels → base_channels
        self.in_conv = ConvBlock(in_channels, base_channels, batchnorm=batchnorm)
        enc_channels.append(base_channels)

        # For each level i=1..depth, do a Down→ConvBlock doubling channels each time:
        curr_ch = base_channels
        self.downs = nn.ModuleList()
        for i in range(depth):
            next_ch = curr_ch * 2
            self.downs.append(Down(curr_ch, next_ch))
            enc_channels.append(next_ch)
            curr_ch = next_ch

        # 2) Bottleneck: from the deepest enc_channels[-1] → 2× that
        bottleneck_in = enc_channels[-1]
        bottleneck_out = bottleneck_in * 2
        self.bottleneck = ConvBlock(bottleneck_in, bottleneck_out, batchnorm=batchnorm)

        # 3) Build the “decoder” (Up blocks).  We’ll mirror `enc_channels` (in reverse)
        #    Each Up takes: current_channels → skip_channels, then ConvBlock(2×skip_channels → skip_channels).
        self.ups = nn.ModuleList()
        curr_ch = bottleneck_out
        for skip_ch in reversed(enc_channels):
            self.ups.append(Up(curr_ch, skip_ch, batchnorm=batchnorm))
            curr_ch = skip_ch

        # 4) Final 1×1 conv: from “base_channels” → out_channels
        #    After the last Up, curr_ch == enc_channels[0] == base_channels.
        self.out_conv = nn.Conv2d(curr_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, in_channels, H, W)

        # ---- Encoder path ----
        enc_feats = []

        # Stage 0 (no pooling)
        x0 = self.in_conv(x)
        enc_feats.append(x0)

        # Stages 1..depth
        x_cur = x0
        for down in self.downs:
            x_cur = down(x_cur)
            enc_feats.append(x_cur)

        # x_cur is now at the “deepest” resolution
        # ---- Bottleneck ----
        x_cur = self.bottleneck(x_cur)

        # ---- Decoder path (use reversed enc_feats) ----
        for up_block, skip_feat in zip(self.ups, reversed(enc_feats)):
            x_cur = up_block(x_cur, skip_feat)

        # ---- Final 1×1 conv to get desired out_channels ----
        return self.out_conv(x_cur)
