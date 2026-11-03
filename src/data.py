"""FER+ / FER2013 label loading, stratified splitting and augmentation."""
from __future__ import annotations

import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import EMOTIONS, RAW_DIR

cv2.setNumThreads(1)


def _decode(blob: bytes) -> np.ndarray:
    buf = np.frombuffer(blob, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")
    if img.shape != (48, 48):
        img = cv2.resize(img, (48, 48), interpolation=cv2.INTER_AREA)
    return img


def load_split(
    split: str, raw_dir: Path = RAW_DIR, label_set: str = "ferplus"
) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 images [N,48,48] and int64 labels for ``train`` or ``test``.

    ``label_set="fer2013"`` uses the original crowd labels shipped with the
    pickle files.  ``label_set="ferplus"`` uses the FER+ ten-vote consensus
    labels derived by ``scripts/build_ferplus_labels.py``; images whose FER+
    majority is ``contempt`` (outside the seven-class protocol) are dropped.
    """
    path = raw_dir / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `bash scripts/download_data.sh` first."
        )
    with open(path, "rb") as fh:
        rows = pickle.load(fh)
    images = np.empty((len(rows), 48, 48), dtype=np.uint8)
    labels = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        images[i] = _decode(row["img_bytes"])
        labels[i] = EMOTIONS.index(row["labels"])

    if label_set == "fer2013":
        return images, labels
    if label_set != "ferplus":
        raise ValueError(f"Unknown label_set {label_set!r}; use 'fer2013' or 'ferplus'.")

    plus_path = raw_dir / "ferplus_labels.npz"
    if not plus_path.exists():
        raise FileNotFoundError(
            f"{plus_path} not found. Run `python scripts/build_ferplus_labels.py` first."
        )
    plus = np.load(plus_path)[f"{split}_labels"]
    keep = plus >= 0
    return images[keep], plus[keep]


def stratified_split(
    labels: np.ndarray, val_fraction: float = 0.10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced train/validation index split."""
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_fraction)))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx), np.asarray(val_idx)


class FERDataset(Dataset):
    """In-memory FER+ / FER2013 subset with optional geometric/photometric augmentation."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray | None = None,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(images)) if indices is None else indices
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        rng = self.rng
        if rng.random() < 0.5:
            img = img[:, ::-1]

        h, w = img.shape
        angle = rng.uniform(-12, 12)
        scale = rng.uniform(0.90, 1.10)
        tx = rng.uniform(-0.07, 0.07) * w
        ty = rng.uniform(-0.07, 0.07) * h
        matrix = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, scale)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        img = cv2.warpAffine(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        alpha = rng.uniform(0.80, 1.20)
        beta = rng.uniform(-22, 22)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if rng.random() < 0.25:
            eh = rng.randint(6, 16)
            ew = rng.randint(6, 16)
            ey = rng.randint(0, h - eh)
            ex = rng.randint(0, w - ew)
            img = img.copy()
            img[ey : ey + eh, ex : ex + ew] = int(img.mean())
        return img

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img = self.images[idx]
        if self.augment:
            img = self._augment(img)
        x = torch.from_numpy(np.ascontiguousarray(img)).float().div_(255.0).sub_(0.5).div_(0.5)
        return x.unsqueeze(0), int(self.labels[idx])


def make_loaders(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images: np.ndarray,
    val_labels: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 8,
    seed: int = 42,
):
    train_ds = FERDataset(train_images, train_labels, augment=True, seed=seed)
    val_ds = FERDataset(val_images, val_labels, augment=False)
    g = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        generator=g,
        persistent_workers=num_workers > 0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader


def denormalize(x: torch.Tensor) -> np.ndarray:
    """[-1,1] tensor [1,48,48] -> uint8 image."""
    arr = (x.detach().cpu().squeeze().numpy() * 0.5 + 0.5) * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


class IndexedSubset(Dataset):
    """Small helper that exposes (image, label, original_index)."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices=None):
        self.images = images
        self.labels = labels
        self.indices = np.arange(len(images)) if indices is None else indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        img = self.images[idx]
        x = torch.from_numpy(img).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
        return x, int(self.labels[idx]), idx
