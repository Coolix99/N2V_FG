# ===== File: src/n2v_fg/apply.py =====

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from n2v_fg.unet import UNet2D


def _extract_patch(
    volume: torch.Tensor,
    t0: int, y0: int, x0: int,
    t_patch: int, y_patch: int, x_patch: int
) -> torch.Tensor:
    """
    Helper: Extract a sub-volume of shape (t_patch, Z, C, y_patch, x_patch)
    from `volume` (shape (T, Z, C, Y, X)) starting at indices (t0, y0, x0).
    """
    # volume: (T, Z, C, Y, X)
    return volume[
        t0 : t0 + t_patch,
        :,
        :,
        y0 : y0 + y_patch,
        x0 : x0 + x_patch
    ]  # → (t_patch, Z, C, y_patch, x_patch)


def _flatten_tzc_to_channels(patch_5d: torch.Tensor) -> torch.Tensor:
    """
    Flatten a 5D patch (t_patch, Z, C, y_patch, x_patch) to a 3D tensor
    (C_total, y_patch, x_patch), where C_total = t_patch * Z * C.
    """
    t_patch, Z, C, H, W = patch_5d.shape
    return patch_5d.contiguous().view(t_patch * Z * C, H, W)


def _unflatten_channels_to_tzc(
    patch_3d: torch.Tensor,
    t_patch: int, Z: int, C: int
) -> torch.Tensor:
    """
    Un-flatten a 3D tensor (C_total, y_patch, x_patch) back to
    a 5D patch (t_patch, Z, C, y_patch, x_patch).
    """
    C_total, H, W = patch_3d.shape
    assert C_total == t_patch * Z * C, f"C_total ({C_total}) != t_patch*Z*C ({t_patch*Z*C})"
    return patch_3d.view(t_patch, Z, C, H, W)


def _apply_tta_and_predict(
    net: UNet2D,
    inp_patch: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Perform Test-Time Augmentation (TTA) on a single patch. We apply:
      - 90° rotations k=0..3 in the XY plane
      - horizontal flip (X), vertical flip (Y)
    For each transform, we run the network, invert the transform on the output,
    and average results.  

    Input:
      inp_patch: torch.Tensor of shape (C_total, y_patch, x_patch)
    Returns:
      out_patch: torch.Tensor of shape (C_total, y_patch, x_patch) (averaged over TTA)
    """
    transforms = []
    # No rotation, no flip
    transforms.append(lambda x: x)
    # Rotations by k*90 degrees
    for k in (1, 2, 3):
        transforms.append(lambda x, k=k: torch.rot90(x, k, dims=(-2, -1)))
    # Horizontal flip
    transforms.append(lambda x: x.flip(dims=[-1]))
    # Vertical flip
    transforms.append(lambda x: x.flip(dims=[-2]))

    inv_transforms = []
    inv_transforms.append(lambda x: x)  # identity
    for k in (1, 2, 3):
        inv_transforms.append(lambda x, k=k: torch.rot90(x, -k, dims=(-2, -1)))
    inv_transforms.append(lambda x: x.flip(dims=[-1]))  # horizontal flip is its own inverse
    inv_transforms.append(lambda x: x.flip(dims=[-2]))  # vertical flip is its own inverse

    preds = []
    for tf, inv_tf in zip(transforms, inv_transforms):
        aug_in = tf(inp_patch)
        aug_in = aug_in.unsqueeze(0).to(device)  # add batch dim
        with torch.no_grad():
            aug_out = net(aug_in)
        aug_out = aug_out.squeeze(0).cpu()
        # invert transform
        aug_out = inv_tf(aug_out)
        preds.append(aug_out)

    # Average all predictions
    stacked = torch.stack(preds, dim=0)  # (n_tta, C_total, H, W)
    return stacked.mean(dim=0)


def apply_network(
    model: Union[UNet2D, torch.nn.Module, str],
    volume: Union[np.ndarray, torch.Tensor],
    patch_size: Tuple[int, int, int] = (8, 128, 128),
    stride: Tuple[int, int, int] = (1, 32, 32),
    device: Union[str, torch.device] = "cpu",
    tta: bool = False
) -> np.ndarray:
    """
    Apply a trained UNet2D to a 5D volume (T, Z, C, Y, X) in overlapping patches
    so that the output has exactly the same shape (T, Z, C, Y, X).

    Arguments:
        model: Either a UNet2D instance (already loaded on `device`) or
               a string path to a `.pth` checkpoint containing `model.state_dict()`.
        volume: 5D array or tensor of shape (T, Z, C, Y, X). Will be converted to float32.
        patch_size: (t_patch, y_patch, x_patch) for the sliding window.
        stride:     (t_stride, y_stride, x_stride) for how far to move the window each step.
                    Overlaps are created when stride < patch_size.
        device: "cpu" or "cuda" (or torch.device).
        tta:     If True, use test-time augmentation (rotations and flips in XY) for each patch.

    Returns:
        out_volume: NumPy array of shape (T, Z, C, Y, X) containing the network’s output,
                    reconstructed by averaging overlapping patches.
    """
    # Ensure model is a UNet2D on the correct device
    if isinstance(model, str):
        # load checkpoint
        checkpoint = torch.load(model, map_location="cpu")
        # We need to know in_channels / out_channels to instantiate the same architecture.
        # Here, assume they saved in the checkpoint a small dict with keys "state_dict" and "arch_kwargs".
        if "arch_kwargs" in checkpoint:
            arch_kwargs = checkpoint["arch_kwargs"]
            net = UNet2D(**arch_kwargs).to(device)
            net.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError("Checkpoint must contain 'arch_kwargs' and 'state_dict'.")
    else:
        net = model.to(device)
    net.eval()

    # Convert input volume to torch.Tensor on CPU
    if isinstance(volume, np.ndarray):
        vol = torch.from_numpy(volume).float()
    else:
        vol = volume.float()
    # vol shape: (T, Z, C, Y, X)
    assert vol.ndim == 5, f"Volume must be 5D (T, Z, C, Y, X), got {vol.shape}"
    T, Z, C, Y, X = vol.shape
    t_patch, y_patch, x_patch = patch_size
    t_stride, y_stride, x_stride = stride

    # Prepare output accumulators
    # We will sum predictions into `accum_vol` and keep counts in `count_vol`
    accum_vol = torch.zeros_like(vol)
    count_vol = torch.zeros_like(vol)

    # Loop over patch coordinates in T, Y, X
    t_positions = list(range(0, T - t_patch + 1, t_stride))
    if t_positions[-1] != T - t_patch:
        t_positions.append(T - t_patch)
    y_positions = list(range(0, Y - y_patch + 1, y_stride))
    if y_positions[-1] != Y - y_patch:
        y_positions.append(Y - y_patch)
    x_positions = list(range(0, X - x_patch + 1, x_stride))
    if x_positions[-1] != X - x_patch:
        x_positions.append(X - x_patch)

    # For progress bar
    total_patches = len(t_positions) * len(y_positions) * len(x_positions)
    pbar = tqdm(total=total_patches, desc="Applying patches", unit="patch")

    # Iterate
    for t0 in t_positions:
        for y0 in y_positions:
            for x0 in x_positions:
                # 1) Extract patch (t_patch, Z, C, y_patch, x_patch)
                patch5 = _extract_patch(vol, t0, y0, x0, t_patch, y_patch, x_patch)
                # 2) Flatten (t_patch, Z, C)→C_total
                patch3 = _flatten_tzc_to_channels(patch5)

                # 3) Move to device and possibly run TTA
                if tta:
                    pred3 = _apply_tta_and_predict(net, patch3, torch.device(device))
                else:
                    inp = patch3.unsqueeze(0).to(device)  # (1, C_total, H, W)
                    with torch.no_grad():
                        out = net(inp)  # (1, C_total, H, W)
                    pred3 = out.squeeze(0).cpu()  # (C_total, H, W)

                # 4) Unflatten back to (t_patch, Z, C, y_patch, x_patch)
                pred5 = _unflatten_channels_to_tzc(pred3, t_patch, Z, C)

                # 5) Add to accum_vol and increment count_vol
                accum_vol[
                    t0 : t0 + t_patch,
                    :,
                    :,
                    y0 : y0 + y_patch,
                    x0 : x0 + x_patch
                ] += pred5
                count_vol[
                    t0 : t0 + t_patch,
                    :,
                    :,
                    y0 : y0 + y_patch,
                    x0 : x0 + x_patch
                ] += 1

                pbar.update(1)

    pbar.close()

    # Avoid division by zero (shouldn’t happen if patch_size ≤ volume dims)
    mask_zero = count_vol == 0
    count_vol[mask_zero] = 1.0

    # Compute final output by averaging
    out_vol = accum_vol / count_vol

    # Convert to NumPy and return
    return out_vol.cpu().numpy()


if __name__ == "__main__":
    """
    Example usage as a script:

        python -m n2v_fg.apply \
            --model_path trained_unet.pth \
            --input_tiff input_volume.tif \
            --output_tiff denoised_volume.tif \
            --patch_size 8 128 128 \
            --stride 4 64 64 \
            --device cuda \
            --tta

    The script will:
      1) Load a TIFF (2D, 3D, or 4D) and reshape/expand to 5D (T,Z,C,Y,X).
      2) Call apply_network(...) to get a denoised 5D output.
      3) Squeeze back to original dimensions and save as a TIFF.
    """

    import argparse
    import tifffile

    parser = argparse.ArgumentParser(description="Apply a trained UNet2D to a TIFF volume")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pth checkpoint containing {'arch_kwargs', 'state_dict'}.")
    parser.add_argument("--input_tiff", type=str, required=True,
                        help="Path to input TIFF (can be 2D, 3D, or 4D).")
    parser.add_argument("--output_tiff", type=str, required=True,
                        help="Path where the output TIFF will be saved.")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[8, 128, 128],
                        help="Patch size (t_patch, y_patch, x_patch).")
    parser.add_argument("--stride", nargs=3, type=int, default=[4, 64, 64],
                        help="Stride (t_stride, y_stride, x_stride).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device to use (e.g. 'cpu' or 'cuda').")
    parser.add_argument("--tta", action="store_true",
                        help="If set, apply test-time augmentation (rotations/flips).")
    args = parser.parse_args()

    # 1) Load the TIFF as NumPy. tifffile can handle 2D, 3D, or 4D.
    img = tifffile.imread(args.input_tiff)
    orig_shape = img.shape
    # Determine how to expand to (T, Z, C, Y, X):
    if img.ndim == 2:
        # (Y, X) → treat as (T=1, Z=1, C=1, Y, X)
        T, Z, C, Y, X = 1, 1, 1, orig_shape[0], orig_shape[1]
        vol5d = img.reshape(T, Z, C, Y, X)
    elif img.ndim == 3:
        # Could be (T, Y, X) or (Z, Y, X). We assume (T, Y, X).
        T, Y, X = orig_shape
        Z, C = 1, 1
        vol5d = img.reshape(T, Z, C, Y, X)
    elif img.ndim == 4:
        # Could be (T, Z, Y, X) or (T, C, Y, X). We assume (T, Z, Y, X) with C=1.
        T, Z, Y, X = orig_shape
        C = 1
        vol5d = img.reshape(T, Z, C, Y, X)
    elif img.ndim == 5:
        vol5d = img  # already (T, Z, C, Y, X)
    else:
        raise ValueError(f"Unsupported input TIFF dims: {orig_shape}")

    # Normalize volume to [0,1] float32
    vol5d = vol5d.astype(np.float32)
    vol5d = (vol5d - vol5d.min()) / (vol5d.max() - vol5d.min() + 1e-8)

    # 2) Apply the network
    denoised_5d = apply_network(
        model=args.model_path,
        volume=vol5d,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        device=args.device,
        tta=args.tta
    )

    # 3) Squeeze back to original dims and save as TIFF
    if img.ndim == 2:
        out_img = denoised_5d.reshape(orig_shape)
    elif img.ndim == 3:
        out_img = denoised_5d.reshape(orig_shape)
    elif img.ndim == 4:
        out_img = denoised_5d.reshape(orig_shape)
    else:  # ndim == 5
        out_img = denoised_5d

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_tiff)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Save
    # Convert back to original dtype/range if desired; here we save float32
    tifffile.imwrite(args.output_tiff, out_img)
    print(f"Saved output TIFF to: {args.output_tiff}")
