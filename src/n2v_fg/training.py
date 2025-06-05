import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from n2v_fg.dataset import Noise2VoidDataset3D
from n2v_fg.net import SpatioTemporalDenoiser


def train_noise2void(
    volumes: List[Union[np.ndarray, torch.Tensor]],
    patch_size: Tuple[int, int, int, int] = (8, 4, 128, 128),
    p_mask: float = 0.01,
    rotate_xy: bool = True,
    flip_xy: bool = True,
    flip_t: bool = True,
    flip_z: bool = False,
    k_t: int = 3,
    gn_channels_per_group: int = 8,
    batch_size: int = 4,
    lr: float = 1e-3,
    num_epochs: int = 20,
    device: Union[str, torch.device] = "cpu",
    val_split: float = 0.1,
    val_volumes: Optional[List[Union[np.ndarray, torch.Tensor]]] = None,
    save_path: Optional[str] = None,
) -> SpatioTemporalDenoiser:
    """
    Train the SpatioTemporalDenoiser in a Noise2Void fashion on 4D volumes of shape (T, Z, Y, X).

    Steps:
      1. Build Noise2VoidDataset3D for train (and optional validation) from `volumes`.
      2. Instantiate DataLoaders, SpatioTemporalDenoiser, optimizer, and masked‐MSE criterion.
      3. Run training loop for `num_epochs`, logging train/val loss.
      4. Optionally save the final model (with arch_kwargs + state_dict) to `save_path`.

    Args:
        volumes: List of 4D arrays or tensors, each shape (T, Z, Y, X).
        patch_size: (T_patch, Z_patch, Y_patch, X_patch) for cropping during training.
        p_mask: fraction of voxels to mask out (~1% by default).
        rotate_xy, flip_xy, flip_t, flip_z: augmentation flags.
        k_t: temporal kernel size for the network’s 1D conv along T.
        gn_channels_per_group: how many channels per group in GroupNorm.
        batch_size: number of patches per batch.
        lr: Adam learning rate.
        num_epochs: number of epochs to train.
        device: “cpu” or “cuda” or torch.device.
        val_split: fraction of `volumes` to hold out for validation (if `val_volumes` is None).
        val_volumes: optionally provide a separate list of volumes for validation.
        save_path: if provided, save trained model to this path (includes arch_kwargs).
    Returns:
        The trained SpatioTemporalDenoiser model (on the appropriate device).
    """
    # 1) Setup logging
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Noise2Void training (4D).")

    # 2) Split volumes into train/val if needed
    if val_volumes is None:
        num_vols = len(volumes)
        num_val = int(round(num_vols * val_split))
        num_train = num_vols - num_val

        if num_train < 1 or num_val < 1:
            logger.warning(
                f"Not enough volumes ({num_vols}) to split into train/val with val_split={val_split}. "
                "Continuing without validation; all volumes used for training."
            )
            train_vols = volumes
            val_vols = []
            do_validation = False
        else:
            # Simple split: first `num_train` → train, remainder → val
            train_vols = volumes[:num_train]
            val_vols = volumes[num_train:]
            do_validation = True
            logger.info(
                f"Split {num_vols} volumes → {len(train_vols)} train / {len(val_vols)} val."
            )
    else:
        train_vols = volumes
        val_vols = val_volumes
        do_validation = True
        logger.info(
            f"Using {len(train_vols)} volumes for training and {len(val_vols)} for validation."
        )

    # 3) Create training dataset & loader
    train_dataset = Noise2VoidDataset3D(
        volumes=train_vols,
        patch_size=patch_size,
        p_mask=p_mask,
        rotate_xy=rotate_xy,
        flip_xy=flip_xy,
        flip_t=flip_t,
        flip_z=flip_z,
        device=None,  # keep data on CPU; move batch to device in loop
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device != "cpu"),
    )

    # 4) Create validation dataset & loader (if applicable)
    if do_validation and len(val_vols) > 0:
        val_dataset = Noise2VoidDataset3D(
            volumes=val_vols,
            patch_size=patch_size,
            p_mask=p_mask,
            rotate_xy=False,  # no augmentation during validation
            flip_xy=False,
            flip_t=False,
            flip_z=False,
            device=None,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(device != "cpu"),
        )
    else:
        val_loader = None
        if not do_validation:
            logger.info("Skipping validation (all data used for training).")

    # 5) Inspect one example volume to extract (T0, Z0, Y0, X0)
    example_vol = train_vols[0]
    if isinstance(example_vol, np.ndarray):
        example_vol = torch.from_numpy(example_vol)
    T0, Z0, Y0, X0 = example_vol.shape

    # 6) Instantiate the SpatioTemporalDenoiser
    #    - num_z = Z_patch
    #    - Use k_t, gn_channels_per_group
    _, Zp, _, _ = patch_size  # Z_patch
    model = SpatioTemporalDenoiser(
        num_z=Zp,
        k_t=k_t,
        gn_channels_per_group=gn_channels_per_group
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 7) Define masked‐MSE loss (3D)
    def masked_mse_loss(
        pred: torch.Tensor,  # shape: (B, T_patch, Z_patch, Y_patch, X_patch)
        target: torch.Tensor,  # same shape
        mask: torch.Tensor      # bool, same shape
    ) -> torch.Tensor:
        diff2 = (pred - target) ** 2
        mask_f = mask.float()  # 1.0 where True
        sum_sq = (diff2 * mask_f).sum()
        n_masked = mask_f.sum()
        if n_masked < 1.0:
            return torch.tensor(0.0, device=diff2.device, requires_grad=False)
        return sum_sq / n_masked

    # 8) Training + Validation Loop
    for epoch in range(1, num_epochs + 1):
        # ------ TRAIN ------
        model.train()
        running_loss = 0.0
        n_batches = 0

        for (masked_patch, orig_patch, mask) in train_loader:
            # masked_patch, orig_patch, mask all are torch Tensors, shape:
            #   (batch_size, T_patch, Z_patch, Y_patch, X_patch)
            masked_patch = masked_patch.to(device)   # FloatTensor
            orig_patch = orig_patch.to(device)
            mask = mask.to(device)                   # BoolTensor

            optimizer.zero_grad()
            output = model(masked_patch)  # → (B, Tp, Zp, Yp, Xp)

            loss = masked_mse_loss(output, orig_patch, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        avg_train_loss = running_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch}/{num_epochs} — Train Loss: {avg_train_loss:.6f}")

        # ----- VALIDATION -----
        if val_loader is not None:
            model.eval()
            val_loss_total = 0.0
            val_batches = 0
            with torch.no_grad():
                for (masked_patch, orig_patch, mask) in val_loader:
                    masked_patch = masked_patch.to(device)
                    orig_patch = orig_patch.to(device)
                    mask = mask.to(device)

                    output = model(masked_patch)
                    loss = masked_mse_loss(output, orig_patch, mask)
                    val_loss_total += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_total / max(val_batches, 1)
            logger.info(f"Epoch {epoch}/{num_epochs} — Val   Loss: {avg_val_loss:.6f}")

    # 9) Save the final model (including arch_kwargs for reloading)
    if save_path is not None:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        arch_kwargs = {
            "num_z": Zp,
            "k_t": k_t,
            "gn_channels_per_group": gn_channels_per_group
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
    
    python -m n2v_fg.training \
      --train_volumes vol1.npy vol2.npy \
      --val_volumes vol3.npy \
      --epochs 10 --batch_size 4 --lr 1e-3 \
      --patch_size 8 4 128 128 --p_mask 0.01 \
      --k_t 5 --gn_group 8 \
      --device cuda \
      --save_path model_checkpoint.pth
    """

    import argparse

    parser = argparse.ArgumentParser(description="Train SpatioTemporalDenoiser in Noise2Void mode")
    parser.add_argument(
        "--train_volumes",
        nargs="+",
        help="Paths to .npy or .pt files containing volumes of shape (T,Z,Y,X).",
        required=True,
    )
    parser.add_argument(
        "--val_volumes",
        nargs="*",
        default=[],
        help="(Optional) Paths to validation .npy or .pt volumes. If omitted, splits train set.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--patch_size",
        nargs=4,
        type=int,
        default=[8, 4, 128, 128],
        help="Patch size: (T_patch, Z_patch, Y_patch, X_patch).",
    )
    parser.add_argument("--p_mask", type=float, default=0.01)
    parser.add_argument(
        "--rotate_xy",
        action="store_true",
        help="Apply random 90° rotation in XY plane.",
    )
    parser.add_argument(
        "--flip_xy",
        action="store_true",
        help="Apply random flips in X and/or Y.",
    )
    parser.add_argument(
        "--flip_t",
        action="store_true",
        help="Apply random flip in T.",
    )
    parser.add_argument(
        "--flip_z",
        action="store_true",
        help="Apply random flip in Z.",
    )
    parser.add_argument(
        "--k_t",
        type=int,
        default=3,
        help="Temporal kernel size for 1D conv along T.",
    )
    parser.add_argument(
        "--gn_group",
        type=int,
        default=8,
        help="Approximate channels per group for GroupNorm.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save_path", type=str, default="trained_stdenoiser.pth")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="If no val_volumes given, fraction of train set held for validation.",
    )

    args = parser.parse_args()

    # Helper to load a 4D volume from .npy or .pt:
    def _load_volume(path: str) -> np.ndarray:
        if path.endswith(".npy"):
            return np.load(path)
        elif path.endswith(".pt"):
            return torch.load(path).cpu().numpy()
        else:
            raise ValueError(f"Unsupported format: {path}")

    train_vs = [_load_volume(p) for p in args.train_volumes]
    val_vs = [_load_volume(p) for p in args.val_volumes] if args.val_volumes else None

    train_noise2void(
        volumes=train_vs,
        val_volumes=val_vs,
        patch_size=tuple(args.patch_size),
        p_mask=args.p_mask,
        rotate_xy=args.rotate_xy,
        flip_xy=args.flip_xy,
        flip_t=args.flip_t,
        flip_z=args.flip_z,
        k_t=args.k_t,
        gn_channels_per_group=args.gn_group,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        device=args.device,
        val_split=args.val_split,
        save_path=args.save_path,
    )
