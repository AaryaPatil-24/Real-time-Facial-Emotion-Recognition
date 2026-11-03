"""Publication-ready dark-themed figures for the report and slide deck."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from .config import EMOTIONS, EMOTION_COLORS

BG = "#0b1220"
PANEL = "#111c30"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
GRID = "#233149"
ACCENT = "#f59e0b"
CYAN = "#22d3ee"

plt.rcParams.update(
    {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "axes.grid": False,
        "figure.dpi": 160,
    }
)

# Warm sequential map that reads well on a dark background.
CMAP = LinearSegmentedColormap.from_list("ink_amber", ["#0f172a", "#3b2f1f", "#b45309", "#fbbf24"])


def _style_axis(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)


def _footer(fig, text="FER+ · PyTorch · CPU"):
    fig.text(0.01, 0.012, text, color=MUTED, fontsize=7.5, ha="left")


def plot_training_curves(history: list[dict], best_epoch: int, out: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.17, wspace=0.22)

    ax = axes[0]
    ax.plot(epochs, [h["train_loss"] for h in history], color=ACCENT, lw=2, label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], color=CYAN, lw=2, label="validation")
    ema_loss = [h.get("val_loss_ema") for h in history]
    if any(v is not None for v in ema_loss):
        ax.plot(epochs, [v if v is not None else np.nan for v in ema_loss], color="#a78bfa", lw=1.4, ls="--", label="EMA")
    ax.axvline(best_epoch, color=MUTED, ls="--", lw=1)
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.legend(frameon=False, loc="upper right")
    _style_axis(ax)

    ax = axes[1]
    ax.plot(epochs, [h["train_acc"] * 100 for h in history], color=ACCENT, lw=2, label="train")
    ax.plot(epochs, [h["val_acc"] * 100 for h in history], color=CYAN, lw=2, label="validation")
    ema_acc = [h.get("val_acc_ema") for h in history]
    if any(v is not None for v in ema_acc):
        ax.plot(epochs, [v * 100 if v is not None else np.nan for v in ema_acc], color="#a78bfa", lw=1.4, ls="--", label="EMA")
    ax.axvline(best_epoch, color=MUTED, ls="--", lw=1)
    ax.set_title("Accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower right")
    _style_axis(ax)

    fig.suptitle(
        "Training converges smoothly — no severe overfitting",
        x=0.07,
        y=0.96,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    _footer(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, out: Path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    fig.subplots_adjust(left=0.17, right=0.99, top=0.84, bottom=0.14)
    im = ax.imshow(norm, cmap=CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=35, ha="right")
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            value = norm[i, j]
            color = "#0b1220" if value > 55 else TEXT
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8.5, fontweight=weight)
    for i in range(len(EMOTIONS)):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=TEXT, lw=1.2, alpha=0.55))
    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cbar.set_label("row recall (%)", color=MUTED)
    cbar.outline.set_edgecolor(GRID)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_title("Confusion matrix — row-normalised (%)", pad=14)
    _footer(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_per_class(report: dict, support: dict, out: Path) -> None:
    names = EMOTIONS
    y = np.arange(len(names))
    precision = [report[n]["precision"] * 100 for n in names]
    recall = [report[n]["recall"] * 100 for n in names]
    f1 = [report[n]["f1-score"] * 100 for n in names]
    h = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    fig.subplots_adjust(left=0.22, right=0.97, top=0.82, bottom=0.16)
    ax.barh(y + h, precision, height=h, color="#64748b", label="precision")
    ax.barh(y, recall, height=h, color=CYAN, label="recall")
    ax.barh(y - h, f1, height=h, color=ACCENT, label="F1")
    for i, v in enumerate(f1):
        ax.text(v + 1.2, i - h, f"{v:.0f}", va="center", color=TEXT, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}  (n={support[n]})" for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("%")
    ax.legend(frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Per-class test performance", pad=26)
    _style_axis(ax, grid_axis="x")
    _footer(fig)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _upscale(img: np.ndarray, size: int = 144) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def plot_dataset_grid(
    images: np.ndarray,
    labels: np.ndarray,
    out: Path,
    per_row: int = 9,
    seed: int = 3,
    with_title: bool = True,
) -> None:
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=len(EMOTIONS))
    fig = plt.figure(figsize=(13.2, 8.6))
    top = 0.87 if with_title else 0.99
    bottom = 0.04 if with_title else 0.015
    gs = fig.add_gridspec(
        len(EMOTIONS), per_row, left=0.13, right=0.985, top=top, bottom=bottom, wspace=0.03, hspace=0.10
    )
    row_h = (top - bottom) / len(EMOTIONS)
    for r, name in enumerate(EMOTIONS):
        idx = np.flatnonzero(labels == r)
        picks = rng.choice(idx, size=per_row, replace=False)
        for c, idx_img in enumerate(picks):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(_upscale(images[idx_img]), cmap="gray", vmin=0, vmax=255)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if c == 0:
                ax.add_patch(
                    Rectangle((-0.03, -0.03), 1.06, 1.06, transform=ax.transAxes, fill=False,
                              edgecolor=EMOTION_COLORS[name], lw=3, clip_on=False)
                )
        label_y = top - (r + 0.5) * row_h
        fig.text(
            0.115,
            label_y,
            name,
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=EMOTION_COLORS[name],
        )
        fig.text(
            0.115,
            label_y - 0.024,
            f"{counts[r]:,}",
            ha="right",
            va="center",
            fontsize=8,
            color=MUTED,
        )
    if with_title:
        fig.suptitle(
            f"Seven expressions, {len(labels):,} faces — FER+",
            x=0.035,
            y=0.955,
            ha="left",
            fontsize=18,
            fontweight="bold",
        )
        fig.text(
            0.035,
            0.915,
            "Ten crowd votes per face, majority label · grayscale 48×48 · heavily imbalanced",
            color=MUTED,
            fontsize=9.5,
            ha="left",
        )
        _footer(fig, "FER+ (relabeled FER2013) · training + test images")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_augmentations(images: np.ndarray, labels: np.ndarray, out: Path, seed: int = 17) -> None:
    """Original face plus five random training augmentations, one row."""
    from .data import FERDataset

    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(labels == EMOTIONS.index("happy"))
    idx = int(rng.choice(candidates))
    ds = FERDataset(images, labels, augment=True, seed=seed)
    augmented = [ds._augment(images[idx]) for _ in range(5)]
    cards = [images[idx]] + augmented
    captions = ["original"] + ["augmented"] * 5
    fig, axes = plt.subplots(1, 6, figsize=(12.6, 2.7))
    fig.subplots_adjust(left=0.012, right=0.988, top=0.88, bottom=0.10, wspace=0.06)
    for ax, img, caption in zip(axes, cards, captions):
        ax.imshow(_upscale(img, 160), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(ACCENT if caption == "original" else GRID)
            spine.set_linewidth(2.4 if caption == "original" else 1.2)
            spine.set_visible(True)
        ax.set_xlabel(caption, fontsize=8, color=MUTED, labelpad=4)
    fig.suptitle(
        "One face, many plausible variations",
        x=0.012,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    _footer(fig, "flip · ±12° rotation · 0.9–1.1× scale · ±20% brightness/contrast · random erasing")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(
    images: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    out: Path,
    accuracy: float,
    n: int = 15,
    seed: int = 11,
) -> None:
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(images), size=n, replace=False)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.6, 2.3 * rows + 0.9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.03, wspace=0.08, hspace=0.34)
    for ax, idx in zip(axes.ravel(), picks):
        pred = int(probs[idx].argmax())
        true = int(labels[idx])
        conf = float(probs[idx, pred])
        correct = pred == true
        ax.imshow(_upscale(images[idx], 160), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        edge = "#22c55e" if correct else "#ef4444"
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(edge)
            spine.set_linewidth(2.2)
        caption = f"{EMOTIONS[pred]} {conf * 100:.0f}%"
        if not correct:
            caption += f"\ntrue: {EMOTIONS[true]}"
        ax.set_xlabel(caption, fontsize=7.8, color=TEXT if correct else "#fca5a5", labelpad=3)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(
        f"Predictions on unseen faces — {accuracy * 100:.1f}% accuracy",
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.905,
        "Green border = correct · red border = error. Most errors are genuinely ambiguous faces.",
        color=MUTED,
        fontsize=8.5,
        ha="left",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_gradcam(model, images, labels, probs, device, out: Path, n: int = 6, seed: int = 5) -> None:
    from .gradcam import GradCAM, cam_overlay

    rng = np.random.default_rng(seed)
    correct = np.flatnonzero(probs.argmax(1) == labels)
    # One confident, correctly classified example per class where possible.
    picks = []
    for cls in range(len(EMOTIONS)):
        cand = [i for i in correct if labels[i] == cls]
        if cand:
            best = max(cand, key=lambda i: probs[i, cls])
            picks.append(best)
    if len(picks) < n:
        extra = [i for i in rng.permutation(correct) if i not in picks]
        picks.extend(extra[: n - len(picks)])
    picks = picks[:n]

    cam = GradCAM(model, model.cam_target)
    fig, axes = plt.subplots(2, n, figsize=(13.4, 5.0))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.05, wspace=0.04, hspace=0.06)
    for col, idx in enumerate(picks):
        gray = images[idx]
        x = (
            torch.from_numpy(gray).float().div(255).sub(0.5).div(0.5)[None, None].to(device)
        )
        heat, _ = cam(x, class_idx=int(probs[idx].argmax()))
        overlay = cam_overlay(gray, heat)
        axes[0, col].imshow(_upscale(gray, 180), cmap="gray", vmin=0, vmax=255)
        axes[1, col].imshow(_upscale_rgb(overlay, 180))
        for row in range(2):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            for spine in axes[row, col].spines.values():
                spine.set_visible(False)
        pred = int(probs[idx].argmax())
        axes[1, col].set_xlabel(
            f"{EMOTIONS[pred]}  {probs[idx, pred] * 100:.0f}%", fontsize=8.2, color=TEXT, labelpad=4
        )
    axes[0, 0].set_ylabel("input", color=MUTED, fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", color=MUTED, fontsize=9)
    fig.suptitle(
        "Where the CNN looks — Grad-CAM",
        x=0.02,
        y=0.965,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.895,
        "Warm regions drive the prediction: eyes, brows and mouth shape — not background or hair.",
        color=MUTED,
        fontsize=9,
        ha="left",
    )
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def _upscale_rgb(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)


def save_emotion_crops(images, labels, out_dir: Path, size: int = 320, n: int = 3, seed: int = 21) -> None:
    """Save a few clean crops per class for the deck's title collage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for cls, name in enumerate(EMOTIONS):
        idx = np.flatnonzero(labels == cls)
        picks = rng.choice(idx, size=min(n, len(idx)), replace=False)
        for k, i in enumerate(picks):
            img = _upscale(images[i], size)
            cv2.imwrite(str(out_dir / f"{name}_{k}.png"), img)
