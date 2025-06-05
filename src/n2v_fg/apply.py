# ===== File: src/n2v_fg/apply.py =====

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from n2v_fg.net import SpatioTemporalDenoiser


def _extract_patch(
    volume: torch.Tensor,
    t0: int, y0: int, x0: int,
    t_patch: int, y_patch: int, x_patch: int
) -> torch.Tensor:
    """
    Helper: Extract a sub‐volume of shape (t_patch, Z, C, y_patch, x_patch)
    from `volume` (shape (T, Z, C, Y, X)) starting at indices (t0, y0, x0).
    """
    return volume[
        t0 : t0 + t_patch,
        :,
        :,
        y0 : y0 + y_patch,
        x0 : x0 + x_patch
    ]  # → (t_patch, Z, C, y_patch, x_patch)


def apply_network(
    model: Union[SpatioTemporalDenoiser, torch.nn.Module, str],
    volume: Union[np.ndarray, torch.Tensor],
    patch_size: Tuple[int, int, int] = (8, 128, 128),
    stride: Tuple[int, int, int]     = (1, 32, 32),
    device: Union[str, torch.device] = "cpu",
    tta: bool = False
) -> np.ndarray:
    """
    Apply a trained SpatioTemporalDenoiser to a 4D or 5D volume in overlapping patches,
    cropping out margins in T and (Y,X) to avoid seam artifacts. The final output
    has the same dimensionality as the input.

    Arguments:
        model: SpatioTemporalDenoiser instance (on `device`) or path to a checkpoint
               (.pth) containing {'arch_kwargs','state_dict'}.
        volume: 4D array/tensor (T,Z,Y,X) or 5D (T,Z,C,Y,X). Will be converted to float32.
        patch_size: (t_patch, y_patch, x_patch) for the sliding window.
        stride:     (t_stride, y_stride, x_stride) defining overlap.
        device:     "cpu" or "cuda" (or torch.device).
        tta:        If True, apply test-time augmentation per patch.

    Returns:
        out_volume: NumPy array of the same shape as the input volume.
    """
    # --------------------------------------------------------------------------
    # 1) Load or receive the network onto the correct device
    # --------------------------------------------------------------------------
    if isinstance(model, str):
        checkpoint = torch.load(model, map_location="cpu")
        if "arch_kwargs" in checkpoint and "state_dict" in checkpoint:
            arch_kwargs = checkpoint["arch_kwargs"]
            net = SpatioTemporalDenoiser(**arch_kwargs).to(device)
            net.load_state_dict(checkpoint["state_dict"])
        else:
            raise ValueError("Checkpoint must contain 'arch_kwargs' and 'state_dict'.")
    else:
        net = model.to(device)
    net.eval()

    # --------------------------------------------------------------------------
    # 2) Convert input to CPU tensor and detect 4D vs 5D
    # --------------------------------------------------------------------------
    if isinstance(volume, np.ndarray):
        vol = torch.from_numpy(volume).float()
    else:
        vol = volume.float()

    if vol.ndim == 4:
        # (T, Z, Y, X) → insert C=1 → (T, Z, 1, Y, X)
        vol = vol.unsqueeze(2)
        squeeze_channel = True
    elif vol.ndim == 5:
        # Already (T, Z, C, Y, X)
        squeeze_channel = False
    else:
        raise AssertionError(f"Volume must be 4D (T,Z,Y,X) or 5D (T,Z,C,Y,X), got {vol.shape}")

    T, Z, C, Y, X = vol.shape
    t_patch, y_patch, x_patch = patch_size
    t_stride, y_stride, x_stride = stride

    # --------------------------------------------------------------------------
    # 3) Decide margins in T, Y, and X (we use ~1/8th of each patch size)
    # --------------------------------------------------------------------------
    margin_t = max(1, t_patch // 8)
    margin_y = max(1, y_patch // 8)
    margin_x = max(1, x_patch // 8)

    # --------------------------------------------------------------------------
    # 4) Prepare accumulators (on CPU)
    # --------------------------------------------------------------------------
    accum_vol = torch.zeros_like(vol)
    count_vol = torch.zeros_like(vol)

    # --------------------------------------------------------------------------
    # 5) Build lists of starting positions
    # --------------------------------------------------------------------------
    t_positions = list(range(0, T - t_patch + 1, t_stride))
    if t_positions[-1] != T - t_patch:
        t_positions.append(T - t_patch)

    y_positions = list(range(0, Y - y_patch + 1, y_stride))
    if y_positions[-1] != Y - y_patch:
        y_positions.append(Y - y_patch)

    x_positions = list(range(0, X - x_patch + 1, x_stride))
    if x_positions[-1] != X - x_patch:
        x_positions.append(X - x_patch)

    total_patches = len(t_positions) * len(y_positions) * len(x_positions)
    pbar = tqdm(total=total_patches, desc="Applying patches", unit="patch")

    # --------------------------------------------------------------------------
    # 6) Slide over every patch
    # --------------------------------------------------------------------------
    for t0 in t_positions:
        for y0 in y_positions:
            for x0 in x_positions:
                # 6.1) Extract raw patch: (t_patch, Z, C, y_patch, x_patch)
                patch5 = _extract_patch(vol, t0, y0, x0, t_patch, y_patch, x_patch)

                # 6.2) Remove the channel dimension C (must be 1) → get (t_patch, Z, y_patch, x_patch)
                patch4 = patch5.squeeze(2)  # now shape = (t_patch, Z, y_patch, x_patch)

                # 6.3) Run the network (with optional TTA)
                if tta:
                    # For simplicity, we only show the “no‐TTA” path here.
                    # If TTA is desired, you would need to rotate/flip the 4D tensor,
                    # feed it into net, invert the transform on the output, and average.
                    raise NotImplementedError("TTA for SpatioTemporalDenoiser not shown")
                else:
                    # net expects (N, T, Z, Y, X):
                    inp = patch4.unsqueeze(0).to(device)  # shape = (1, t_patch, Z, y_patch, x_patch)
                    with torch.no_grad():
                        out4d = net(inp)               # shape = (1, t_patch, Z, y_patch, x_patch)
                    pred4 = out4d.squeeze(0).cpu()     # shape = (t_patch, Z, y_patch, x_patch)

                # 6.4) Re‐insert channel dimension C=1 → (t_patch, Z, 1, y_patch, x_patch)
                pred5 = pred4.unsqueeze(2)

                # 6.5) Compute crop‐ranges in T, Y, and X
                t1, t2 = margin_t, t_patch - margin_t
                y1, y2 = margin_y, y_patch - margin_y
                x1, x2 = margin_x, x_patch - margin_x

                # If patch is too small in any axis, skip cropping that axis
                if t2 <= t1:
                    cropped_t0, cropped_t1 = 0, t_patch
                else:
                    cropped_t0, cropped_t1 = t1, t2

                if y2 <= y1:
                    cropped_y0, cropped_y1 = 0, y_patch
                else:
                    cropped_y0, cropped_y1 = y1, y2

                if x2 <= x1:
                    cropped_x0, cropped_x1 = 0, x_patch
                else:
                    cropped_x0, cropped_x1 = x1, x2

                # Extract the cropped block from pred5:
                #  → cropped5 shape = (t2−t1, Z, 1, y2−y1, x2−x1) (or full extents if no crop)
                cropped5 = pred5[
                    cropped_t0:cropped_t1,
                    :,
                    :,
                    cropped_y0:cropped_y1,
                    cropped_x0:cropped_x1
                ]

                # 6.6) Accumulate into accum_vol & increment count_vol
                accum_vol[
                    t0 + cropped_t0 : t0 + cropped_t1,
                    :,
                    :,
                    y0 + cropped_y0 : y0 + cropped_y1,
                    x0 + cropped_x0 : x0 + cropped_x1
                ] += cropped5

                count_vol[
                    t0 + cropped_t0 : t0 + cropped_t1,
                    :,
                    :,
                    y0 + cropped_y0 : y0 + cropped_y1,
                    x0 + cropped_x0 : x0 + cropped_x1
                ] += 1

                pbar.update(1)

    pbar.close()

    # --------------------------------------------------------------------------
    # 7) Finalize: divide by counts (avoid zeros)
    # --------------------------------------------------------------------------
    zero_mask = (count_vol == 0)
    count_vol[zero_mask] = 1.0
    out_vol = accum_vol / count_vol

    # --------------------------------------------------------------------------
    # 8) If input was 4D, squeeze away channel dim to return a 4D result again
    # --------------------------------------------------------------------------
    if squeeze_channel:
        out_vol = out_vol.squeeze(2)  # (T, Z, Y, X)

    # --------------------------------------------------------------------------
    # 9) Return as NumPy
    # --------------------------------------------------------------------------
    return out_vol.cpu().numpy()


if __name__ == "__main__":
    """
    Example usage as a script:

        python -m n2v_fg.apply \
            --model_path trained_stdenoiser.pth \
            --input_tiff input_volume.tif \
            --output_tiff denoised_volume.tif \
            --patch_size 8 128 128 \
            --stride 4 64 64 \
            --device cuda \
            --tta

    The script will:
      1) Load a TIFF (2D/3D/4D), expand to 5D (T,Z,C,Y,X).
      2) Call apply_network() to get a denoised 5D output.
      3) Squeeze back to original dims and save as a TIFF.
    """

    import argparse
    import tifffile

    parser = argparse.ArgumentParser(description="Apply a trained SpatioTemporalDenoiser to a TIFF volume")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to .pth checkpoint containing {'arch_kwargs', 'state_dict'}.")
    parser.add_argument("--input_tiff", type=str, required=True,
                        help="Path to input TIFF (2D, 3D, or 4D).")
    parser.add_argument("--output_tiff", type=str, required=True,
                        help="Path where the output TIFF will be saved.")
    parser.add_argument("--patch_size", nargs=3, type=int, default=[8, 128, 128],
                        help="Patch size (t_patch, y_patch, x_patch).")
    parser.add_argument("--stride", nargs=3, type=int, default=[4, 64, 64],
                        help="Stride (t_stride, y_stride, x_stride).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device to use (e.g. 'cpu' or 'cuda').")
    parser.add_argument("--tta", action="store_true",
                        help="If set, apply test‐time augmentation.")
    args = parser.parse_args()

    # 1) Load the TIFF into a NumPy array
    img = tifffile.imread(args.input_tiff)
    orig_shape = img.shape

    # 2) Expand to (T, Z, C, Y, X)
    if img.ndim == 2:
        # (Y, X) → (T=1, Z=1, C=1, Y, X)
        T, Z, C, Y, X = 1, 1, 1, orig_shape[0], orig_shape[1]
        vol5d = img.reshape(T, Z, C, Y, X)
    elif img.ndim == 3:
        # (T, Y, X) → (T, Z=1, C=1, Y, X)
        T, Y, X = orig_shape
        Z, C = 1, 1
        vol5d = img.reshape(T, Z, C, Y, X)
    elif img.ndim == 4:
        # (T, Z, Y, X) → (T, Z, C=1, Y, X)
        T, Z, Y, X = orig_shape
        C = 1
        vol5d = img.reshape(T, Z, C, Y, X)
    elif img.ndim == 5:
        vol5d = img  # already (T, Z, C, Y, X)
    else:
        raise ValueError(f"Unsupported input TIFF dims: {orig_shape}")

    # 3) Normalize to [0,1] float32
    vol5d = vol5d.astype(np.float32)
    vol5d = (vol5d - vol5d.min()) / (vol5d.max() - vol5d.min() + 1e-8)

    # 4) Apply the network
    denoised_vol = apply_network(
        model=args.model_path,
        volume=vol5d,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.stride),
        device=args.device,
        tta=args.tta
    )

    # 5) Squeeze back and save
    if img.ndim == 2:
        out_img = denoised_vol.reshape(orig_shape)
    elif img.ndim == 3:
        out_img = denoised_vol.reshape(orig_shape)
    elif img.ndim == 4:
        out_img = denoised_vol.reshape(orig_shape)
    else:  # ndim == 5
        out_img = denoised_vol

    # 6) Ensure output directory exists
    out_dir = os.path.dirname(args.output_tiff)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 7) Save as float32 TIFF
    tifffile.imwrite(args.output_tiff, out_img)
    print(f"Saved output TIFF to: {args.output_tiff}")
