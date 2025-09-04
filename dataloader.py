import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional

# =========================================================
# Utility functions
# =========================================================

def extract_flowrate_from_filename(filename: str) -> Optional[float]:
    """Extract the first floating-point number from filename."""
    match = re.search(r"([\d.]+)", filename.lower())
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def load_video_frames(video_path: str, target_resolution: Tuple[int, int]) -> List[np.ndarray]:
    """Load grayscale frames from video file and resize to target resolution."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[ERROR] Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"[ERROR] Cannot open video: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, target_resolution, interpolation=cv2.INTER_AREA)
        frames.append(gray.astype(np.float32))

    cap.release()
    if not frames:
        raise RuntimeError(f"[ERROR] No frames found in: {video_path}")

    return frames

# =========================================================
# Dataset class
# =========================================================

class SpeckleDataset(Dataset):
    """PyTorch Dataset for 2D CNN speckle videos."""
    def __init__(
        self,
        stacks: np.ndarray,
        labels: np.ndarray,
        normalize: str = "zscore",
        train: bool = True,
        augment: bool = False
    ):
        assert stacks.ndim == 5, f"Expected stacks shape (N,1,T,H,W), got {stacks.shape}"
        self.x = stacks.astype(np.float32)
        self.y = labels.astype(np.float32)
        self.normalize = normalize
        self.train = train
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        clip = self.x[idx]  # (1,T,H,W)
        label = self.y[idx]

        # Normalization
        if self.normalize == "scale":
            clip = clip / 255.0
        elif self.normalize == "zscore":
            mu = clip.mean()
            sd = clip.std()
            clip = (clip - mu) / sd if sd > 1e-6 else clip - mu

        clip = torch.from_numpy(clip)

        # Data augmentation
        if self.train and self.augment:
            clip = self._apply_augmentation(clip)

        return clip, torch.tensor(label, dtype=torch.float32)

    def _apply_augmentation(self, clip: torch.Tensor) -> torch.Tensor:
        """Random augmentations: brightness, noise, flips."""
        if np.random.rand() < 0.5:
            clip *= 1.0 + np.random.uniform(-0.1, 0.1)  # brightness
        if np.random.rand() < 0.5:
            clip += torch.randn_like(clip) * 0.01  # Gaussian noise
        if np.random.rand() < 0.5:
            clip = torch.flip(clip, dims=[-1])  # horizontal flip
        if np.random.rand() < 0.5:
            clip = torch.flip(clip, dims=[-2])  # vertical flip
        return clip

# =========================================================
# Dataset preparation
# =========================================================

def prepare_dataset(
    data_folder: str,
    sequence_len: int = 5,
    stride: int = 1,
    cache_file: Optional[str] = None,
    target_resolution: Tuple[int, int] = (128, 128)
) -> Tuple[np.ndarray, np.ndarray]:
    """Create dataset of video stacks (N,1,T,H,W) and corresponding labels."""
    if cache_file and os.path.exists(cache_file):
        print(f"[INFO] Loading cached dataset from: {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached["stacks"], cached["labels"]

    print(f"[INFO] Preparing dataset from: {data_folder}")
    all_stacks, all_labels = [], []

    for filename in sorted(os.listdir(data_folder)):
        if not filename.lower().endswith(".avi"):
            continue

        flowrate = extract_flowrate_from_filename(filename)
        if flowrate is None or not np.isfinite(flowrate) or flowrate < 0:
            print(f"[WARN] Skipping invalid filename or flowrate: {filename}")
            continue

        video_path = os.path.join(data_folder, filename)
        try:
            frames = load_video_frames(video_path, target_resolution)
        except Exception as e:
            print(f"[ERROR] Failed to load {filename}: {e}")
            continue

        if len(frames) < sequence_len:
            print(f"[WARN] Skipping short video ({len(frames)} frames): {filename}")
            continue

        # Create overlapping clips
        for i in range(0, len(frames) - sequence_len + 1, stride):
            clip = np.stack(frames[i:i + sequence_len], axis=0)  # (T,H,W)
            clip = clip[np.newaxis, ...]  # (1,T,H,W)
            all_stacks.append(clip)
            all_labels.append(flowrate)

            # Duplicate samples for large videos (~40MB)
            if os.path.getsize(video_path) > 40_000_000:
                all_stacks.append(clip.copy())
                all_labels.append(flowrate)

    if not all_stacks:
        raise RuntimeError("[ERROR] No valid samples extracted from dataset.")

    stacks = np.stack(all_stacks, axis=0)  # (N,1,T,H,W)
    labels = np.array(all_labels, dtype=np.float32)

    if cache_file:
        print(f"[INFO] Saving dataset cache to: {cache_file}")
        np.savez_compressed(cache_file, stacks=stacks, labels=labels)

    print(f"[INFO] Total samples created: {len(stacks)}")
    return stacks, labels

# =========================================================
# DataLoader creation
# =========================================================

def create_dataloaders(
    data_folder: str,
    batch_size: int = 8,
    test_split: float = 0.2,
    sequence_len: int = 5,
    stride: int = 1,
    use_subset: bool = False,
    max_samples: int = 300,
    val_sample_size: Optional[int] = None,
    normalize_mode: str = "zscore",
    num_workers: int = 0,
    seed: int = 42,
    cache_file: Optional[str] = None,
    augment_train: bool = True,
    safe_mode: bool = True,
    target_resolution: Tuple[int, int] = (128, 128)
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Create PyTorch DataLoaders for train/validation."""
    stacks, labels = prepare_dataset(
        data_folder=data_folder,
        sequence_len=sequence_len,
        stride=stride,
        cache_file=cache_file,
        target_resolution=target_resolution
    )

    rng = np.random.default_rng(seed)

    if use_subset and len(stacks) > max_samples:
        idx = rng.permutation(len(stacks))[:max_samples]
        stacks, labels = stacks[idx], labels[idx]

    if len(stacks) < 2:
        raise ValueError("[ERROR] Not enough samples to split dataset.")

    X_train, X_val, y_train, y_val = train_test_split(
        stacks, labels, test_size=test_split, random_state=seed, shuffle=True
    )

    if val_sample_size and len(X_val) > val_sample_size:
        idx = rng.permutation(len(X_val))[:val_sample_size]
        X_val, y_val = X_val[idx], y_val[idx]

    train_set = SpeckleDataset(X_train, y_train, normalize=normalize_mode, train=True, augment=augment_train)
    val_set = SpeckleDataset(X_val, y_val, normalize=normalize_mode, train=False, augment=False)

    if safe_mode:
        num_workers = 0  # Prevents Windows deadlocks

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, None
