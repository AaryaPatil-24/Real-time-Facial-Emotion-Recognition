"""Shared inference helpers (checkpoint loading, face preprocessing, TTA)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import EMOTIONS
from .model import build_model


def load_model(
    checkpoint: str | Path = "models/emotion_cnn_ferplus.pt",
    device: str | torch.device = "cpu",
) -> tuple[nn.Module, dict]:
    device = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(
        arch=ckpt.get("arch", "cnn"),
        num_classes=len(ckpt.get("classes", EMOTIONS)),
        dropout=float(ckpt.get("dropout", 0.4)),
        width=int(ckpt.get("width", 32)),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    if device.type == "cpu":
        model = model.to(memory_format=torch.channels_last)
    return model, ckpt


def preprocess_face(gray: np.ndarray, use_clahe: bool = False) -> torch.Tensor:
    """uint8 grayscale face -> normalized [1,1,48,48] tensor."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.shape != (48, 48):
        gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    if use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    arr = gray.astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def predict_face(
    model: nn.Module,
    gray: np.ndarray,
    device: str | torch.device = "cpu",
    tta: bool = True,
    use_clahe: bool = False,
) -> np.ndarray:
    device = torch.device(device)
    x = preprocess_face(gray, use_clahe).to(device)
    if device.type == "cpu":
        x = x.contiguous(memory_format=torch.channels_last)
    probs = F.softmax(model(x), dim=1)
    if tta:
        probs = probs + F.softmax(model(torch.flip(x, dims=[3])), dim=1)
        probs = probs / 2.0
    return probs[0].cpu().numpy()


class FaceDetector:
    """Thin wrapper around OpenCV's bundled Haar cascade."""

    def __init__(self, min_size: int = 56, scale_factor: float = 1.1, min_neighbors: int = 5):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError(
                "This OpenCV build has no Haar cascade support. "
                "Install OpenCV 4.x: pip install 'opencv-python>=4.9,<5' "
                "(or 'opencv-python-headless>=4.9,<5' on a server)."
            )
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade at {path}")
        self.min_size = min_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        faces = [tuple(map(int, f)) for f in faces]
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)


def crop_face(frame_bgr: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.12):
    x, y, w, h = box
    dx, dy = int(w * margin), int(h * margin)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1 = min(frame_bgr.shape[1], x + w + dx)
    y1 = min(frame_bgr.shape[0], y + h + dy)
    crop = frame_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.size == 0:
        return np.zeros((48, 48), np.uint8), (x0, y0, x1 - x0, y1 - y0)
    return cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA), (x0, y0, x1 - x0, y1 - y0)
