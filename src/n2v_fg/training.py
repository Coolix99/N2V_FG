# ===== File: src/n2v_fg/training.py =====

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from n2v_fg.dataset import Noise2VoidDataset
from n2v_fg.unet import UNet2D


def train_noise2void(
    volumes: List[Union[np.ndarray, torch.Tensor]],
    patch_size: Tuple[int, int, int] = (8, 128, 128),
    p_mask: float = 0.01,
    rotate: bool = True,
    flip_xy: bool = True,
    flip_t: bool = True,
    base_channels: int = 32,
    depth: int = 2,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_epochs: int = 20,
    device: Union[str, torch.device] = "cpu",
    val_split: float = 0.1,
    val_volumes: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    save_path: Optional[str] = None,
) -> UNet2D:
    """
    Train a UNet2D in a Noise2Void fashion on 5D volumes of shape (T, Z, C, Y, X).

    This function will:
      1. Create a Noise2VoidDataset from `volumes`. If `val_volumes` is None, it will
         attempt to split a fraction `val_split` out of `volumes` for validation. 
         If there aren’t enough volumes, it will warn and proceed without validation.
      2. Instantiate DataLoaders for training (and, if possible, validation).
      3. Create a UNet2D model, optimizer, and masked‐MSE criterion.
      4. Run a standard training loop over `num_epochs` epochs, logging train/val losses.
      5. Optionally save the final model weights to `save_path`.

    Arguments:
        volumes: List of volumes (each is either a NumPy array or a torch.Tensor of shape (T, Z, C, Y, X)).
        patch_size: Tuple (t_patch, y_patch, x_patch) for cropping during training.
        p_mask: Fraction of pixels to mask out in each patch (e.g. 0.01 → 1% of pixels).
        rotate: Whether to apply random 90° rotations in the XY plane.
        flip_xy: Whether to apply random horizontal/vertical flips in the XY plane.
        flip_t: Whether to apply random flips along T.
        base_channels: Number of base feature channels in the U-Net.
        depth: Number of down/up‐sampling levels in the U-Net.
        batch_size: Batch size for both training and validation loaders.
        lr: Adam learning rate.
        num_epochs: Number of epochs to train.
        device: "cpu" or "cuda" or torch.device.
        val_split: If `val_volumes` is None, this fraction of `volumes` will be held out for validation.
        val_volumes: If provided, use these volumes (instead of splitting) to build a validation set.
        save_path: If provided, will save `model.state_dict()` to this path at the end of training.

    Returns:
        The trained UNet2D model (on CPU if device="cpu", or on GPU if device="cuda").
    """
    # ------------------------
    # 1) Setup logging
    # ------------------------
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Noise2Void training.")

    # ------------------------
    # 2) Prepare datasets
    # ------------------------
    if val_volumes is None:
        num_vols = len(volumes)
        num_val = int(round(num_vols * val_split))
        num_train = num_vols - num_val

        if num_train < 1 or num_val < 1:
            logger.warning(
                f"Not enough volumes ({num_vols}) to split into train/val with val_split={val_split}. "
                "Continuing without validation; all volumes will be used for training."
            )
            train_vols = volumes
            val_vols = []
            do_validation = False
        else:
            train_vols = volumes[:num_train]
            val_vols = volumes[num_train:]
            do_validation = True
            logger.info(f"Split {num_vols} volumes into {len(train_vols)} train / {len(val_vols)} val.")
    else:
        train_vols = volumes
        val_vols = val_volumes
        do_validation = True
        logger.info(f"Using {len(train_vols)} volumes for training and {len(val_vols)} for validation.")

    # Create training dataset & loader (dataset always on CPU)
    train_dataset = Noise2VoidDataset(
        volumes=train_vols,
        patch_size=patch_size,
        p_mask=p_mask,
        rotate=rotate,
        flip_xy=flip_xy,
        flip_t=flip_t,
        device=None,  # keep everything on CPU in workers
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

    # Create validation dataset & loader (if any)
    if do_validation and len(val_vols) > 0:
        val_dataset = Noise2VoidDataset(
            volumes=val_vols,
            patch_size=patch_size,
            p_mask=p_mask,
            rotate=False,
            flip_xy=False,
            flip_t=False,
            device=None,  # CPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=False,
        )
    else:
        val_loader = None
        if not do_validation:
            logger.info("Skipping validation (all data used for training).")

    # ------------------------
    # 3) Build model, optimizer, criterion
    # ------------------------
    example_vol = train_vols[0]
    if isinstance(example_vol, np.ndarray):
        example_vol = torch.from_numpy(example_vol)
    T0, Z0, C0, _, _ = example_vol.shape
    in_channels = patch_size[0] * Z0 * C0
    out_channels = in_channels

    # Instantiate U-Net
    model = UNet2D(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        depth=depth,
        batchnorm=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE only over masked pixels.
        pred, target: (B, C_total, H, W)
        mask: BoolTensor same shape, True = pixel was masked.
        """
        diff2 = (pred - target) ** 2
        mask_f = mask.float()  # 1.0 where True
        sum_sq = (diff2 * mask_f).sum()
        n_masked = mask_f.sum()
        if n_masked < 1.0:
            return torch.tensor(0.0, device=diff2.device, requires_grad=False)
        return sum_sq / n_masked

    # ------------------------
    # 4) Training Loop (and optional Validation)
    # ------------------------
    for epoch in range(1, num_epochs + 1):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        n_batches = 0

        for masked_in, orig_patch, mask in train_loader:
            # Move batch to device here
            masked_in = masked_in.to(device)
            orig_patch = orig_patch.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            output = model(masked_in)
            loss = masked_mse_loss(output, orig_patch, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_train_loss:.6f}")

        # ---- VALIDATION (if any) ----
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for masked_in, orig_patch, mask in val_loader:
                    masked_in = masked_in.to(device)
                    orig_patch = orig_patch.to(device)
                    mask = mask.to(device)

                    output = model(masked_in)
                    loss = masked_mse_loss(output, orig_patch, mask)
                    val_loss_total += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_total / max(val_batches, 1)
            logger.info(f"Epoch {epoch}/{num_epochs} — Val   Loss: {avg_val_loss:.6f}")

    # ------------------------
    # 5) Save the final model (only based on training)
    # ------------------------
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Build the arch_kwargs dict so that apply.py can re‐instantiate UNet2D
        arch_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "base_channels": base_channels,
            "depth": depth,
            "batchnorm": True,
        }
        torch.save(
            {
                "arch_kwargs": arch_kwargs,
                "state_dict": model.state_dict()
            },
            save_path
        )
        logger.info(f"Saved trained model (with arch_kwargs) to: {save_path}")

    logger.info("Training complete.")
    return model

if __name__ == "__main__":
    """
    Example usage as a script. Adjust the paths and hyperparameters below.
    
    python -m n2v_fg.training

    This code snippet will:
      - Load two example volumes from .npy files
      - Train for 10 epochs on small patches (for demonstration).
      - Save the model to "trained_unet.pth" in the current directory.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Train UNet2D in Noise2Void mode")
    parser.add_argument(
        "--train_volumes",
        nargs="+",
        help="Paths to .npy or .pt files containing volumes of shape (T,Z,C,Y,X).",
        required=True,
    )
    parser.add_argument(
        "--val_volumes",
        nargs="*",
        default=[],
        help="(Optional) Paths to validation .npy or .pt volumes. If omitted, do an internal split.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[8, 128, 128])
    parser.add_argument("--p_mask", type=float, default=0.01)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="trained_unet.pth")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="If no val_volumes given, fraction of train volumes to hold out for validation.",
    )

    args = parser.parse_args()

    # Helper to load a volume from .npy or .pt
    def _load_volume(path: str) -> np.ndarray:
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".pt"):
            return torch.load(path).cpu().numpy()
        else:
            raise ValueError(f"Unsupported volume format: {path}")

    train_vs = [_load_volume(p) for p in args.train_volumes]
    val_vs = [_load_volume(p) for p in args.val_volumes] if args.val_volumes else None

    train_noise2void(
        volumes=train_vs,
        val_volumes=val_vs,
        patch_size=tuple(args.patch_size),
        p_mask=args.p_mask,
        rotate=True,
        flip_xy=True,
        flip_t=True,
        base_channels=args.base_channels,
        depth=args.depth,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        val_split=args.val_split,
        save_path=args.save_path,
    )
