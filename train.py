import os
import sys
import time
import argparse
import logging
from typing import Tuple, List, Optional

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch import amp
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda.amp.autocast")

from src.dataloader import create_dataloaders
from models.bloodflow_cnn import get_model

# =========================================================
# Utilities
# =========================================================

def set_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(output_dir: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    logger = logging.getLogger("bloodflow_train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
        fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger

def atomic_save_scaler(scaler: StandardScaler, path: str, logger: Optional[logging.Logger] = None) -> None:
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "wb") as f:
            joblib.dump(scaler, f)
        os.replace(temp_path, path)
        if logger:
            logger.info(f"Scaler saved to {path}")
    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        if logger:
            logger.error(f"Failed to save scaler to {path}: {e}")
        raise

def save_predictions_csv(preds: List[float], targets: List[float], out_path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        mse = mean_squared_error(targets, preds) if preds else float("nan")
        mae = mean_absolute_error(targets, preds) if preds else float("nan")
        r2 = r2_score(targets, preds) if preds else float("nan")
        w.writerow(["Summary"])
        w.writerow(["MSE", f"{mse:.6f}"])
        w.writerow(["MAE", f"{mae:.6f}"])
        w.writerow(["R2", f"{r2:.6f}"])
        w.writerow([])
        w.writerow(["Index", "TrueFlow", "PredictedFlow", "AbsError", "SquaredError", "RelErrorPercent", "Class"])
        for i, (t, p) in enumerate(zip(targets, preds)):
            abs_e = abs(t - p)
            sq = abs_e ** 2
            rel = (abs_e / t * 100.0) if t != 0 else 0.0
            cls = "Low" if t < 20 else ("Medium" if t < 200 else "High")
            w.writerow([i, f"{t:.6f}", f"{p:.6f}", f"{abs_e:.6f}", f"{sq:.6f}", f"{rel:.2f}", cls])

def plot_loss_curve(train_losses: List[float], out_path: str) -> None:
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (MSE)")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_pred_scatter(targets: List[float], preds: List[float], out_path: str) -> None:
    if not targets or not preds:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.7, edgecolors="k")
    mn = min(min(targets), min(preds))
    mx = max(max(targets), max(preds))
    plt.plot([mn, mx], [mn, mx], "r--", label="Ideal")
    plt.xlabel("True Flow (uL/min)")
    plt.ylabel("Predicted Flow (uL/min)")
    plt.title("Predicted vs True")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# =========================================================
# Train / Validate
# =========================================================

def _fix_input_shape(xb: torch.Tensor) -> torch.Tensor:
    """Ensure input is (B, C, H, W) for 2D CNN."""
    if xb.dim() == 5 and xb.size(2) == 1:
        xb = xb.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
    return xb

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: StandardScaler,
    amp_scaler: Optional[GradScaler],
    grad_clip: float,
    logger: Optional[logging.Logger] = None,
) -> float:
    model.train()
    running_loss = 0.0
    total = 0
    for xb, yb in tqdm(loader, desc="train", leave=False, dynamic_ncols=True):
        xb = _fix_input_shape(xb).to(device, non_blocking=True)
        yb_np = yb.cpu().numpy().reshape(-1, 1)
        yb_scaled = torch.tensor(scaler.transform(yb_np).reshape(-1), dtype=torch.float32, device=device)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast("cuda", enabled=(amp_scaler is not None and device.type == "cuda")):
            preds = model(xb).view(-1)
            loss = criterion(preds, yb_scaled)

        if amp_scaler:
            amp_scaler.scale(loss).backward()
            if grad_clip > 0:
                amp_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += loss.item() * xb.size(0)
        total += xb.size(0)

    avg_loss = running_loss / total if total > 0 else float("nan")
    if logger:
        logger.info(f"Train Loss: {avg_loss:.6f}")
    return avg_loss

def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler: StandardScaler,
    amp_scaler: Optional[GradScaler],
) -> Tuple[float, List[float], List[float]]:
    model.eval()
    running_loss = 0.0
    total = 0
    all_preds: List[float] = []
    all_targets: List[float] = []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="val", leave=False, dynamic_ncols=True):
            xb = _fix_input_shape(xb).to(device, non_blocking=True)
            yb_np = yb.cpu().numpy().reshape(-1, 1)
            yb_scaled = torch.tensor(scaler.transform(yb_np).reshape(-1), dtype=torch.float32, device=device)

            with amp.autocast("cuda", enabled=(amp_scaler is not None and device.type == "cuda")):
                out = model(xb).view(-1)
                loss = criterion(out, yb_scaled)

            running_loss += loss.item() * xb.size(0)
            total += xb.size(0)

            out_cpu = out.detach().cpu().numpy().reshape(-1, 1)
            all_preds.extend(scaler.inverse_transform(out_cpu).reshape(-1).tolist())
            all_targets.extend(yb_np.reshape(-1).tolist())

    avg_loss = running_loss / total if total > 0 else float("nan")
    return avg_loss, all_preds, all_targets

# =========================================================
# Main train / evaluate
# =========================================================

def train(
    data_dir,
    output_dir,
    batch_size,
    sequence_len,
    stride,
    num_epochs,
    lr,
    weight_decay,
    patience,
    val_samples,
    test_split,
    num_workers,
    grad_clip,
    seed,
    normalize,
    cache_videos,
    augment_train,
    safe_mode=False,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info("Preparing dataloaders...")

    train_loader, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=test_split,
        stride=stride,
        num_workers=(0 if safe_mode else num_workers),
        normalize_mode=normalize,
        cache_file=None if not cache_videos else os.path.join(output_dir, "dataset_cache.npz"),
        augment_train=augment_train,
        safe_mode=safe_mode,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    logger.info(f"Model created and moved to {device}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4)

    # Fit scaler on training labels
    target_list = [y.cpu().numpy().reshape(-1, 1) for _, y in train_loader]
    if not target_list:
        raise RuntimeError("No training labels found to fit scaler.")
    train_targets = np.concatenate(target_list, axis=0)
    scaler = StandardScaler()
    scaler.fit(train_targets)
    atomic_save_scaler(scaler, os.path.join(output_dir, "scaler.pkl"), logger)

    amp_scaler = GradScaler(enabled=(device.type == "cuda"))
    best_val = float("inf")
    train_losses: List[float] = []
    best_ckpt = os.path.join(checkpoints_dir, "best_model.pth")

    logger.info("Starting training loop...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_scaler, grad_clip, logger)
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device, scaler, amp_scaler)
        train_losses.append(train_loss)

        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {time.time() - t0:.1f}s")
        scheduler.step(val_loss)

        save_predictions_csv(val_preds[:val_samples], val_targets[:val_samples], os.path.join(output_dir, f"predictions_epoch_{epoch}.csv"))

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model.state_dict()}, best_ckpt)
            logger.info(f"Saved best model to {best_ckpt}")

    # Final evaluation
    ck = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    final_loss, final_preds, final_targets = validate(model, val_loader, criterion, device, scaler, amp_scaler)
    logger.info(f"Final metrics | MSE: {mean_squared_error(final_targets, final_preds):.6f} | MAE: {mean_absolute_error(final_targets, final_preds):.6f} | R2: {r2_score(final_targets, final_preds):.6f}")

    save_predictions_csv(final_preds[:val_samples], final_targets[:val_samples], os.path.join(output_dir, "final_predictions.csv"))
    plot_pred_scatter(final_targets, final_preds, os.path.join(output_dir, "pred_vs_true.png"))
    plot_loss_curve(train_losses, os.path.join(output_dir, "training_loss.png"))
    logger.info("Training complete.")

def evaluate(
    checkpoint: str,
    data_dir: str,
    output_dir: str,
    batch_size: int,
    sequence_len: int,
    stride: int,
    num_workers: int,
    normalize: str,
    cache_videos: bool,
):
    logger = setup_logger(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on device: {device}")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    ck = torch.load(checkpoint, map_location=device)
    model = get_model().to(device)
    model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck)
    model.eval()

    scaler_path = os.path.join(output_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise RuntimeError("Scaler not found. Train first or provide scaler.pkl")
    scaler = joblib.load(scaler_path)
    logger.info(f"Loaded scaler from {scaler_path}")

    _, val_loader, _ = create_dataloaders(
        data_folder=data_dir,
        batch_size=batch_size,
        sequence_len=sequence_len,
        test_split=0.2,
        stride=stride,
        num_workers=num_workers,
        normalize_mode=normalize,
        cache_file=None if not cache_videos else os.path.join(output_dir, "dataset_cache.npz"),
        augment_train=False,
    )

    _, preds, targets = validate(model, val_loader, nn.MSELoss(), device, scaler, amp_scaler=None)

    logger.info("Evaluation complete. Saving outputs...")
    save_predictions_csv(preds, targets, os.path.join(output_dir, "eval_predictions.csv"))
    plot_pred_scatter(targets, preds, os.path.join(output_dir, "eval_pred_vs_true.png"))
    logger.info(f"Eval metrics | MSE: {mean_squared_error(targets, preds):.6f} | MAE: {mean_absolute_error(targets, preds):.6f} | R2: {r2_score(targets, preds):.6f}")

# =========================================================
# CLI
# =========================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train or evaluate BloodFlow 2D-CNN")
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--sequence_len", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_samples", type=int, default=120)
    p.add_argument("--test_split", type=float, default=0.2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", choices=["scale", "zscore"], default="scale")
    p.add_argument("--cache_videos", action="store_true")
    p.add_argument("--augment_train", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    return p.parse_args()

def main():
    args = parse_args()
    if args.mode == "train":
        train(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sequence_len=args.sequence_len,
            stride=args.stride,
            num_epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            val_samples=args.val_samples,
            test_split=args.test_split,
            num_workers=args.num_workers,
            grad_clip=args.grad_clip,
            seed=args.seed,
            normalize=args.normalize,
            cache_videos=args.cache_videos,
            augment_train=args.augment_train,
            safe_mode=False
        )
    else:
        if not args.checkpoint:
            raise ValueError("Please provide --checkpoint for eval mode")
        evaluate(
            checkpoint=args.checkpoint,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            sequence_len=args.sequence_len,
            stride=args.stride,
            num_workers=args.num_workers,
            normalize=args.normalize,
            cache_videos=args.cache_videos,
        )

if __name__ == "__main__":
    main()
