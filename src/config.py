"""Central configuration for the facial emotion recognition project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
PRESENTATION_DIR = ROOT / "presentation"
ASSET_DIR = PRESENTATION_DIR / "assets"

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "sad": "😢",
    "surprise": "😲",
    "neutral": "😐",
}
# Colour-blind friendly accents, one per class.
EMOTION_COLORS = {
    "angry": "#ef4444",
    "disgust": "#84cc16",
    "fear": "#a855f7",
    "happy": "#f59e0b",
    "sad": "#3b82f6",
    "surprise": "#06b6d4",
    "neutral": "#94a3b8",
}


@dataclass
class TrainConfig:
    arch: str = "resnet"
    label_set: str = "ferplus"
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.35
    label_smoothing: float = 0.05
    val_fraction: float = 0.10
    seed: int = 42
    num_workers: int = 8
    patience: int = 15
    width: int = 32
    use_class_weights: bool = False
    augment: bool = True
    image_size: int = 48
    checkpoint: Path = MODEL_DIR / "emotion_cnn_ferplus.pt"
    history: Path = REPORT_DIR / "history.json"


@dataclass
class EvalConfig:
    label_set: str = "ferplus"
    batch_size: int = 256
    tta: bool = True
    checkpoint: Path = MODEL_DIR / "emotion_cnn_ferplus.pt"
    metrics: Path = REPORT_DIR / "metrics.json"


TRAIN = TrainConfig()
EVAL = EvalConfig()
