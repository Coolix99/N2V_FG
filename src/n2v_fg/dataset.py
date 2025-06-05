import random
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def apply_random_mask(
    patch: torch.Tensor,
    p_mask: float = 0.01,
    inplace: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly “mask out” a fraction of pixels in `patch` by zeroing them.
    Returns (masked_patch, mask), where:
      - patch:  a FloatTensor of shape (C_total, H, W).
      - p_mask: fraction of pixels to mask (0 < p_mask < 1). 
                If p_mask=0.01, roughly 1% of all pixels in all channels will be masked.
      - inplace: if True, will zero out `patch` in place; otherwise, works on a copy.
      - device:  if provided, the mask will be created on that device.

    The returned `mask` is a BoolTensor of shape (C_total, H, W), where True=“this pixel was masked.” 
    During training you should compute loss only on the masked positions.
    """
    if not inplace:
        patch = patch.clone()

    # total number of pixels across all channels
    C, H, W = patch.shape
    total_pixels = C * H * W
    n_mask = int(round(p_mask * total_pixels))

    # Create a flat index of all pixel positions
    # We'll sample n_mask indices without replacement
    if device is None:
        device = patch.device
    all_inds = torch.randperm(total_pixels, device=device)
    masked_inds = all_inds[:n_mask]

    # Build the boolean mask
    mask = torch.zeros((C * H * W,), dtype=torch.bool, device=device)
    mask[masked_inds] = True
    mask = mask.view(C, H, W)

    # Zero out (mask) those positions
    patch = patch.masked_fill(mask, 0.0)

    return patch, mask


class Noise2VoidDataset(Dataset):
    """
    A PyTorch Dataset for Noise2Void‐style self-supervised training
    on 5D input volumes of shape (T, Z, C, Y, X). This class:
      1. Randomly extracts a patch of shape (t_patch, Z, C, y_patch, x_patch)
         from a random volume (if you passed multiple to `volumes`).
      2. Applies random 90° rotations/flips in the (Y, X) plane, and optional flip in T.
      3. Flattens (T, Z, C) into a single “channel” dimension of size C_total = t_patch * Z * C.
      4. Applies a random mask (zero‐out) on ≈p_mask fraction of pixels in (C_total, Y, X).
      5. Returns (masked_input, original_patch, mask) all of shape (C_total, y_patch, x_patch).

    Example usage:
        dataset = Noise2VoidDataset(
            volumes=[my_tensor1, my_tensor2],            # each is shape (T,Z,C,Y,X)
            patch_size=(8, 128, 128),                     # (t_patch, y_patch, x_patch)
            p_mask=0.01,                                  # mask ~1% of pixels
            rotate=True, flip_xy=True, flip_t=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
        for (inp, orig, m) in loader:
            # inp, orig, m have shape [4, C_total, 128, 128]
            # compute `pred = model(inp)`
            # loss = ((pred - orig)**2 * m).sum() / m.sum()

    Parameters:
        volumes: List of numpy arrays or torch.Tensors, each with shape (T, Z, C, Y, X).
                 If numpy arrays are passed, they are converted to torch.FloatTensor.
        patch_size: (t_patch, y_patch, x_patch). Must be <= actual (T, Y, X) of each volume.
        p_mask: fraction of pixels to mask out in each patch.
        rotate: whether to apply a random 90° rotation around the Z‐axis (XY plane).
        flip_xy: whether to apply random horizontal/vertical flips in the XY plane.
        flip_t: whether to apply random flip along the T axis.
        device: if provided, all outputs will be moved to this device.
    """
    def __init__(
        self,
        volumes: List[Union[np.ndarray, torch.Tensor]],
        patch_size: Tuple[int, int, int],
        p_mask: float = 0.01,
        rotate: bool = True,
        flip_xy: bool = True,
        flip_t: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # Validate patch_size
        assert len(patch_size) == 3, "patch_size must be (t_patch, y_patch, x_patch)."
        self.t_patch, self.y_patch, self.x_patch = patch_size
        self.p_mask = p_mask
        self.rotate = rotate
        self.flip_xy = flip_xy
        self.flip_t = flip_t
        self.device = device

        # Convert all volumes to torch.FloatTensor (shape: T,Z,C,Y,X)
        self.volumes: List[torch.Tensor] = []
        for vol in volumes:
            if isinstance(vol, np.ndarray):
                vol = torch.from_numpy(vol)
            assert isinstance(vol, torch.Tensor), "Volumes must be numpy arrays or torch Tensors."
            # Ensure float32
            vol = vol.float()
            # Check dimensionality
            assert vol.ndim == 5, "Each volume must be 5D (T, Z, C, Y, X)."
            T, Z, C, Y, X = vol.shape
            # Patch must fit
            assert self.t_patch <= T, f"t_patch={self.t_patch} is greater than T={T}."
            assert self.y_patch <= Y, f"y_patch={self.y_patch} is greater than Y={Y}."
            assert self.x_patch <= X, f"x_patch={self.x_patch} is greater than X={X}."
            self.volumes.append(vol)

    def __len__(self) -> int:
        # We just return a large number; typically you'll wrap in a DataLoader with shuffle,
        # so you can sample indefinitely. If you want a fixed number of samples per epoch,
        # return that here instead (e.g. len(self.volumes)*100).
        return len(self.volumes) * 1000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            masked_patch: FloatTensor, shape (C_total, y_patch, x_patch) on self.device
            original_patch: FloatTensor, same shape (C_total, y_patch, x_patch)
            mask: BoolTensor, same shape (C_total, y_patch, x_patch); True=“masked here”
        """
        # 1) Randomly pick one of the volumes
        vol = random.choice(self.volumes)  # vol.shape = (T, Z, C, Y, X)
        T, Z, C, Y, X = vol.shape

        # 2) Randomly choose a starting index for (t, y, x)
        t0 = random.randint(0, T - self.t_patch)
        y0 = random.randint(0, Y - self.y_patch)
        x0 = random.randint(0, X - self.x_patch)

        # 3) Extract the patch: shape (t_patch, Z, C, y_patch, x_patch)
        patch = vol[
            t0 : t0 + self.t_patch,
            :,
            :,
            y0 : y0 + self.y_patch,
            x0 : x0 + self.x_patch,
        ]  # → (t_patch, Z, C, y_patch, x_patch)

        # 4) Data augmentations (on the 5D patch):
        #    → random 90° rotations and flips in (y_patch, x_patch)
        if self.rotate:
            # Choose k ∈ {0,1,2,3}: number of 90° rotations
            k = random.randint(0, 3)
            if k > 0:
                patch = torch.rot90(patch, k, dims=(-2, -1))  # rotate in the Y–X plane

        if self.flip_xy:
            # horizontal flip (flip X) with 50% chance
            if random.random() < 0.5:
                patch = patch.flip(dims=[-1])
            # vertical flip (flip Y) with 50% chance
            if random.random() < 0.5:
                patch = patch.flip(dims=[-2])

        if self.flip_t:
            # flip along T dimension with 50% chance
            if random.random() < 0.5:
                patch = patch.flip(dims=[0])

        # 5) Flatten (T, Z, C) into the “channel” dimension
        #    After this, shape = (C_total, y_patch, x_patch), where
        #      C_total = t_patch * Z * C
        #    We do: patch.permute(0,1,2,3,4) → already in (T, Z, C, Y, X) order,
        #    so we can just reshape if memory is contiguous.
        #    To be safe, call contiguous() first.
        patch = patch.contiguous()
        T2, Z2, C2, H2, W2 = patch.shape  # T2 == self.t_patch, H2 == self.y_patch, W2 == self.x_patch
        C_total = T2 * Z2 * C2
        patch = patch.view(C_total, H2, W2)

        # 6) Prepare “original” (target) copy and apply random masking to get “input”
        if self.device is not None:
            patch = patch.to(self.device)

        original_patch = patch.clone()  # will be used to compute loss on masked pixels

        # 7) Apply random mask
        masked_patch, mask = apply_random_mask(
            patch, p_mask=self.p_mask, inplace=False, device=patch.device
        )

        return masked_patch, original_patch, mask