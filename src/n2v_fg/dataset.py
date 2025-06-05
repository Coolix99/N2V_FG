import random
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


def apply_random_mask_3d(
    patch: torch.Tensor,
    p_mask: float = 0.01,
    inplace: bool = False,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly “mask out” a fraction of voxels in `patch` by zeroing them.
    Returns (masked_patch, mask), where:
      - patch:  a FloatTensor of shape (T_patch, Z, Y_patch, X_patch).
      - p_mask: fraction of voxels to mask (0 < p_mask < 1). 
                If p_mask=0.01, roughly 1% of all voxels in all channels will be masked.
      - inplace: if True, zero out `patch` in place; otherwise, work on a copy.
      - device:  if provided, the mask will be created on that device.

    The returned `mask` is a BoolTensor of shape (T_patch, Z, Y_patch, X_patch), 
    where True = “this voxel was masked.” During training you compute loss only on these voxels.
    """
    if not inplace:
        patch = patch.clone()

    # total number of voxels across (T_patch, Z, Y, X)
    T, Z, Y, X = patch.shape
    total_voxels = T * Z * Y * X
    n_mask = int(round(p_mask * total_voxels))

    # Build a flat permutation of all voxel‐indices
    if device is None:
        device = patch.device
    all_inds = torch.randperm(total_voxels, device=device)
    masked_inds = all_inds[:n_mask]  # these will be masked

    # Create boolean mask and reshape
    mask = torch.zeros((total_voxels,), dtype=torch.bool, device=device)
    mask[masked_inds] = True
    mask = mask.view(T, Z, Y, X)

    # Zero‐out (mask) those voxels
    patch = patch.masked_fill(mask, 0.0)

    return patch, mask


class Noise2VoidDataset3D(Dataset):
    """
    A PyTorch Dataset for Noise2Void‐style self‐supervised training
    on 4D patches of shape (T_patch, Z, Y_patch, X_patch), drawn from
    4D input volumes of shape (T, Z, Y, X).

    Steps for each sampled patch:
      1. Randomly extract a 4D patch (T_patch, Z, Y_patch, X_patch) from one volume.
      2. Apply random 90° rotations/flips in the (Y, X) plane.
      3. Optionally apply random flips in T (temporal) and/or Z (depth) axes.
      4. Return (masked_patch, original_patch, mask), all of shape
         (T_patch, Z, Y_patch, X_patch).

    Example usage:
        dataset = Noise2VoidDataset3D(
            volumes=[vol1, vol2],          # each is shape (T, Z, Y, X)
            patch_size=(8, 4, 128, 128),    # (T_patch, Z, Y_patch, X_patch)
            p_mask=0.01,                    # mask ~1% of voxels
            rotate_xy=True, flip_xy=True, 
            flip_t=True, flip_z=True
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        for masked, orig, mask in loader:
            # masked, orig, mask have shape [2, 8, 4, 128, 128]
            # feed `masked` into your model, compute loss only where mask=True.
    """
    def __init__(
        self,
        volumes: List[Union[np.ndarray, torch.Tensor]],
        patch_size: Tuple[int, int, int, int],
        p_mask: float = 0.01,
        rotate_xy: bool = True,
        flip_xy: bool = True,
        flip_t: bool = True,
        flip_z: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            volumes: List of numpy arrays or torch Tensors, each of shape (T, Z, Y, X).
            patch_size: (T_patch, Z_patch, Y_patch, X_patch). Must be ≤ actual (T, Z, Y, X) of each volume.
            p_mask: fraction of total voxels to mask out in each patch.
            rotate_xy: if True, randomly rotate each patch by k×90° in the Y–X plane.
            flip_xy: if True, randomly flip Y or X axes.
            flip_t: if True, randomly flip along T (temporal) axis.
            flip_z: if True, randomly flip along Z (depth) axis.
            device: if provided, move the output patches to this device; otherwise return CPU tensors.
        """
        super().__init__()
        assert len(patch_size) == 4, "patch_size must be (T_patch, Z_patch, Y_patch, X_patch)."
        self.Tp, self.Zp, self.Yp, self.Xp = patch_size
        self.p_mask = p_mask
        self.rotate_xy = rotate_xy
        self.flip_xy = flip_xy
        self.flip_t = flip_t
        self.flip_z = flip_z
        self.device = device

        # Convert all volumes to torch.FloatTensor
        self.volumes: List[torch.Tensor] = []
        for vol in volumes:
            if isinstance(vol, np.ndarray):
                vol = torch.from_numpy(vol)
            assert isinstance(vol, torch.Tensor), "Volumes must be numpy arrays or torch Tensors."
            vol = vol.float()
            # Check shape
            assert vol.ndim == 4, "Each volume must be 4D (T, Z, Y, X)."
            T, Z, Y, X = vol.shape
            # Ensure patch fits
            assert self.Tp <= T, f"T_patch={self.Tp} > T={T}"
            assert self.Zp <= Z, f"Z_patch={self.Zp} > Z={Z}"
            assert self.Yp <= Y, f"Y_patch={self.Yp} > Y={Y}"
            assert self.Xp <= X, f"X_patch={self.Xp} > X={X}"
            self.volumes.append(vol)

    def __len__(self) -> int:
        # Return large number so that DataLoader can sample indefinitely.
        return len(self.volumes) * 1000

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            masked_patch: FloatTensor, shape (T_patch, Z, Y_patch, X_patch), on self.device if set
            original_patch: FloatTensor, same shape
            mask: BoolTensor, same shape; True = “masked”
        """
        # 1) Pick a random volume
        vol = random.choice(self.volumes)  # shape = (T, Z, Y, X)
        T, Z, Y, X = vol.shape

        # 2) Pick random start indices for (t, z, y, x)
        t0 = random.randint(0, T - self.Tp)
        z0 = random.randint(0, Z - self.Zp)
        y0 = random.randint(0, Y - self.Yp)
        x0 = random.randint(0, X - self.Xp)

        # 3) Extract the 4D patch: (T_patch, Z_patch, Y_patch, X_patch)
        patch = vol[
            t0 : t0 + self.Tp,
            z0 : z0 + self.Zp,
            y0 : y0 + self.Yp,
            x0 : x0 + self.Xp,
        ].clone()  # → (T_patch, Z_patch, Y_patch, X_patch)

        # 4) Data augmentation on this 4D patch:

        # 4.a) Random rotation in Y–X plane (k×90°)
        if self.rotate_xy:
            k = random.randint(0, 3)
            if k > 0:
                patch = torch.rot90(patch, k, dims=(-2, -1))  # rotate last two dims (Y, X)

        # 4.b) Random flip in X and/or Y
        if self.flip_xy:
            if random.random() < 0.5:
                patch = patch.flip(dims=[-1])  # flip X axis
            if random.random() < 0.5:
                patch = patch.flip(dims=[-2])  # flip Y axis

        # 4.c) Random flip in T
        if self.flip_t and random.random() < 0.5:
            patch = patch.flip(dims=[0])    # flip T axis

        # 4.d) Random flip in Z
        if self.flip_z and random.random() < 0.5:
            patch = patch.flip(dims=[1])    # flip Z axis

        # 5) “Original” copy (target)
        if self.device is not None:
            patch = patch.to(self.device)
        original_patch = patch.clone()

        # 6) Apply random 3D mask
        masked_patch, mask = apply_random_mask_3d(
            patch, p_mask=self.p_mask, inplace=False, device=patch.device
        )

        return masked_patch, original_patch, mask
