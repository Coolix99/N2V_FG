import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConvBlock(nn.Module):
    """
    A “temporal” block that first lifts from num_z → hidden_channels via a 3D conv 
    along T (kernel = (k_t,1,1)), then applies ReLU, and projects back to num_z
    with another 3D conv (also kernel = (k_t,1,1)).  GroupNorm or BatchNorm is omitted 
    here, but you could insert it between the two convs if desired.

    Input shape: (N, T, Z, Y, X)
      - We permute to (N, Z, T, Y, X) so that Conv3d’s “channel” dimension is Z
      - First conv3d: (Z) → (hidden_channels), kernel=(k_t,1,1)
      - ReLU, then second conv3d: (hidden_channels) → (Z), kernel=(k_t,1,1)
      - Permute back to (N, T, Z, Y, X)

    Args:
        num_z:  number of slices in Z (i.e. input “channel”)
        hidden_channels: how many features to produce in the hidden layer
        k_t: kernel size along T
    """
    def __init__(self, num_z: int, hidden_channels: int, k_t: int = 3):
        super().__init__()
        self.k_t = k_t

        # First 3D convolution: in_channels=num_z, out_channels=hidden_channels
        self.conv_reduce = nn.Conv3d(
            in_channels=num_z,
            out_channels=hidden_channels,
            kernel_size=(k_t, 1, 1),
            padding=(k_t // 2, 0, 0),
            bias=True
        )
        # Second 3D convolution: in_channels=hidden_channels, out_channels=num_z
        self.conv_expand = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=num_z,
            kernel_size=(k_t, 1, 1),
            padding=(k_t // 2, 0, 0),
            bias=True
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, Z, Y, X)
        N, T, Z, Y, X = x.shape

        # Permute to (N, Z, T, Y, X) so that Z acts as channels for Conv3d
        x = x.permute(0, 2, 1, 3, 4)   # → (N, Z, T, Y, X)
        # Reduce → hidden_channels
        x = self.conv_reduce(x)        # (N, hidden_channels, T, Y, X)
        x = self.relu(x)
        # Expand back → num_z
        x = self.conv_expand(x)        # (N, Z, T, Y, X)
        x = self.relu(x)
        # Permute back to (N, T, Z, Y, X)
        x = x.permute(0, 2, 1, 3, 4)    # → (N, T, Z, Y, X)
        return x


class SpatialConvBlock(nn.Module):
    """
    A “spatial” block that lifts from num_z → hidden_channels via a 2D conv over (Y,X),
    applies GroupNorm + ReLU, then projects back from hidden_channels → num_z with another 2D conv.

    Input shape: (N, T, Z, Y, X)
      - First reshape → (N*T, Z, Y, X)
      - conv2d: (Z) → (hidden_channels); GroupNorm(hidden_channels) → ReLU
      - conv2d: (hidden_channels) → (Z); ReLU
      - reshape back to (N, T, Z, Y, X)

    Args:
        num_z: number of slices in Z (i.e. input “channel”)
        hidden_channels: number of filters in the hidden 2D conv
        gn_channels_per_group: how many channels per group in GroupNorm
    """
    def __init__(self, num_z: int, hidden_channels: int, gn_channels_per_group: int = 8):
        super().__init__()
        # First 2D convolution: in_channels=num_z, out_channels=hidden_channels
        self.conv_reduce = nn.Conv2d(
            in_channels=num_z,
            out_channels=hidden_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        # GroupNorm on hidden_channels
        num_groups = max(1, hidden_channels // gn_channels_per_group)
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=hidden_channels)
        self.relu = nn.ReLU(inplace=True)
        # Second 2D convolution: in_channels=hidden_channels, out_channels=num_z
        self.conv_expand = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=num_z,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T, Z, Y, X)
        N, T, Z, Y, X = x.shape

        # Reshape to (N*T, Z, Y, X)
        x2 = x.reshape(N * T, Z, Y, X)   # (N*T, Z, Y, X)
        # Reduce → hidden_channels
        x2 = self.conv_reduce(x2)        # (N*T, hidden_channels, Y, X)
        x2 = self.gn(x2)
        x2 = self.relu(x2)
        # Expand back → num_z
        x2 = self.conv_expand(x2)        # (N*T, Z, Y, X)
        x2 = self.relu2(x2)
        # Reshape back to (N, T, Z, Y, X)
        x2 = x2.reshape(N, T, Z, Y, X)
        return x2


class SpatioTemporalDenoiser(nn.Module):
    """
    A deeper SpatioTemporalDenoiser that alternates between:
      - TemporalConvBlock → SpatialConvBlock
      - TemporalConvBlock → SpatialConvBlock
      - Final 1×1×1 Conv3d to produce an unbounded output

    Each TemporalConvBlock and SpatialConvBlock uses a “hidden_channels” dimension 
    internally (wider than num_z) to give more capacity.

    Input:  (N, T, Z, Y, X)
    Output: (N, T, Z, Y, X)  (unbounded)

    Args:
        num_z:    number of slices in Z (the output channel dimension)
        hidden_channels: how many feature‐maps to use in the intermediate layers
        k_t:      temporal kernel size for 1D‐like conv along T
        gn_channels_per_group: channels/group for GroupNorm in spatial blocks
    """
    def __init__(
        self,
        num_z: int,
        hidden_channels: int = 32,
        k_t: int = 3,
        gn_channels_per_group: int = 8,
    ):
        super().__init__()
        # First pair of blocks
        self.tblock1 = TemporalConvBlock(num_z=num_z, hidden_channels=hidden_channels, k_t=k_t)
        self.sblock1 = SpatialConvBlock(num_z=num_z, hidden_channels=hidden_channels,
                                        gn_channels_per_group=gn_channels_per_group)
        # Second pair of blocks
        self.tblock2 = TemporalConvBlock(num_z=num_z, hidden_channels=hidden_channels, k_t=k_t)
        self.sblock2 = SpatialConvBlock(num_z=num_z, hidden_channels=hidden_channels,
                                        gn_channels_per_group=gn_channels_per_group)

        # Final “fusion” conv: 1×1×1 Conv3d (no activation → unbounded output)
        self.final_conv = nn.Conv3d(
            in_channels=num_z,
            out_channels=num_z,
            kernel_size=(1, 1, 1),
            bias=True
        )
        self.sigmoid=nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, Z, Y, X)
        Returns:
            out: (N, T, Z, Y, X)
        """
        # (N, T, Z, Y, X) → first temporal, then spatial
        x = self.tblock1(x)   # → (N, T, Z, Y, X)
        x = self.sblock1(x)   # → (N, T, Z, Y, X)

        # second temporal, then spatial
        x = self.tblock2(x)   # → (N, T, Z, Y, X)
        x = self.sblock2(x)   # → (N, T, Z, Y, X)

        # Final 1×1×1 conv across (T,Z,Y,X). Need to permute so that Z is “channel” for Conv3d:
        x = x.permute(0, 2, 1, 3, 4)        # → (N, Z, T, Y, X)
        out = self.final_conv(x)           # → (N, Z, T, Y, X)
        out = out.permute(0, 2, 1, 3, 4)    # → (N, T, Z, Y, X)
        return self.sigmoid(out)
