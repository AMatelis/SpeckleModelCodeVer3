import os
import re
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def extract_flow_from_filename(fname: str) -> Optional[float]:
    """
    Extract numeric flow rate from a filename.
    Handles formats like '5ul.avi', 'flow_5.0.avi', etc.
    Returns None if no valid number is found.
    """
    m = re.search(r"([0-9]+\.?[0-9]*)", fname)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def read_video_frames(video_path: str) -> List[np.ndarray]:
    """
    Read all frames from a video in grayscale.
    Returns a list of float32 numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Could not open video: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame.astype(np.float32))
    cap.release()
    return frames


class Speckle2DDataset(Dataset):
    """
    2D CNN Speckle flow dataset (frame-wise), optional caching.

    Each frame is treated independently (T dimension is removed).

    Args:
        folder (str): Path to folder containing .avi videos.
        stride (int): Step between frames when sampling.
        normalize (str): 'scale' or 'zscore'.
        cache_videos (bool): Cache entire videos in memory.
        shuffle_frames (bool): Shuffle frames within each video.
    """

    def __init__(
        self,
        folder: str,
        stride: int = 1,
        normalize: str = "scale",
        cache_videos: bool = False,
        shuffle_frames: bool = False,
    ):
        self.folder = folder
        self.stride = stride
        self.normalize = normalize
        self.cache_videos = cache_videos
        self.shuffle_frames = shuffle_frames

        self.samples: List[Tuple[str, int, float]] = []
        self.cache: Dict[str, List[np.ndarray]] = {}

        if not os.path.isdir(folder):
            raise NotADirectoryError(f"[ERROR] Dataset folder not found: {folder}")

        files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".avi")])
        if not files:
            raise RuntimeError(f"[ERROR] No .avi files found in {folder}")

        for fn in files:
            flow = extract_flow_from_filename(fn)
            if flow is None:
                print(f"[WARN] Could not parse flow rate from filename: {fn}")
                continue

            path = os.path.join(folder, fn)
            frames = read_video_frames(path)
            if len(frames) == 0:
                print(f"[WARN] No frames found in video: {fn}")
                continue

            if cache_videos:
                self.cache[path] = frames

            indices = list(range(0, len(frames), stride))
            if shuffle_frames:
                np.random.shuffle(indices)

            for i in indices:
                self.samples.append((path, i, float(flow)))

        if not self.samples:
            raise RuntimeError(f"[ERROR] No valid frames found in {folder}")

        print(f"[INFO] Prepared {len(self.samples)} 2D frame samples from {len(files)} videos.")

    def __len__(self) -> int:
        return len(self.samples)

    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Normalize a single 2D frame."""
        if self.normalize == "scale":
            frame = frame / 255.0
        elif self.normalize == "zscore":
            mu, sd = frame.mean(), frame.std()
            frame = (frame - mu) / (sd + 1e-8)
        return frame.astype(np.float32)

    def _get_frame(self, path: str, idx: int) -> torch.Tensor:
        """Retrieve a single frame from cache or disk."""
        if path in self.cache:
            frame = self.cache[path][idx]
        else:
            frames = read_video_frames(path)
            frame = frames[idx]
        frame = self._normalize_frame(frame)
        frame = frame[None, ...]  # Add channel dimension (1, H, W)
        return torch.from_numpy(frame)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, frame_idx, flow = self.samples[idx]
        x = self._get_frame(path, frame_idx)
        y = torch.tensor(flow, dtype=torch.float32)
        return x, y
