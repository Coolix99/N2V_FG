import torch
import numpy as np
from n2v_fg.dataset import apply_random_mask, Noise2VoidDataset


def test_apply_random_mask():
    C, H, W = 3, 32, 32
    p_mask = 0.05
    patch = torch.ones(C, H, W)
    masked_patch, mask = apply_random_mask(patch, p_mask=p_mask)

    assert masked_patch.shape == patch.shape
    assert mask.shape == patch.shape
    assert mask.dtype == torch.bool
    assert torch.all(masked_patch[mask] == 0)
    expected_masked = int(round(p_mask * C * H * W))
    actual_masked = mask.sum().item()
    assert abs(actual_masked - expected_masked) < C * H
    print("✅ test_apply_random_mask passed.")


def test_noise2void_dataset_sample():
    # Small dummy volume: (T=6, Z=1, C=2, Y=64, X=64)
    vol = np.random.rand(6, 1, 2, 64, 64).astype(np.float32)

    dataset = Noise2VoidDataset(
        volumes=[vol],
        patch_size=(3, 32, 32),
        p_mask=0.02,
        rotate=True,
        flip_xy=True,
        flip_t=True,
        device=torch.device("cpu"),
    )

    masked_patch, original_patch, mask = dataset[0]
    C_total = 3 * 1 * 2  # t_patch * Z * C = 6
    assert masked_patch.shape == (C_total, 32, 32)
    assert original_patch.shape == (C_total, 32, 32)
    assert mask.shape == (C_total, 32, 32)
    assert torch.all(masked_patch[mask] == 0)
    assert torch.all(original_patch[mask] != 0)
    print("✅ test_noise2void_dataset_sample passed.")


if __name__ == "__main__":
    test_apply_random_mask()
    test_noise2void_dataset_sample()
