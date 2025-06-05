import torch
import numpy as np
from n2v_fg.dataset import apply_random_mask_3d, Noise2VoidDataset3D  # adjust path if needed


def test_apply_random_mask_3d():
    T, Z, Y, X = 4, 2, 32, 32
    p_mask = 0.05
    patch = torch.ones(T, Z, Y, X)
    masked_patch, mask = apply_random_mask_3d(patch, p_mask=p_mask)

    assert masked_patch.shape == patch.shape
    assert mask.shape == patch.shape
    assert mask.dtype == torch.bool
    assert torch.all(masked_patch[mask] == 0)

    total_voxels = T * Z * Y * X
    expected_masked = int(round(p_mask * total_voxels))
    actual_masked = mask.sum().item()
    assert abs(actual_masked - expected_masked) < Z * Y  # allow a bit of tolerance
    print("✅ test_apply_random_mask_3d passed.")


def test_noise2void_dataset3d_sample():
    # Small dummy volume: (T=6, Z=3, Y=64, X=64)
    vol = np.random.rand(6, 3, 64, 64).astype(np.float32)

    dataset = Noise2VoidDataset3D(
        volumes=[vol],
        patch_size=(4, 2, 32, 32),
        p_mask=0.02,
        rotate_xy=True,
        flip_xy=True,
        flip_t=True,
        flip_z=True,
        device=torch.device("cpu"),
    )

    masked_patch, original_patch, mask = dataset[0]
    assert masked_patch.shape == (4, 2, 32, 32)
    assert original_patch.shape == (4, 2, 32, 32)
    assert mask.shape == (4, 2, 32, 32)
    assert masked_patch.dtype == torch.float32
    assert mask.dtype == torch.bool
    assert torch.all(masked_patch[mask] == 0)
    assert torch.all(original_patch[mask] != 0)
    print("✅ test_noise2void_dataset3d_sample passed.")


if __name__ == "__main__":
    test_apply_random_mask_3d()
    test_noise2void_dataset3d_sample()
