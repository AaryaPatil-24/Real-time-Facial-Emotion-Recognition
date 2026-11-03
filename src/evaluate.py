"""Evaluate the trained CNN on the held-out FER+ (or FER2013) test split.

Produces reports/metrics.json, reports/test_probs.npy and all figures used by
the README and the presentation.

Usage:
    python -m src.evaluate
    python -m src.evaluate --no-tta --no-figures
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .config import (
    ASSET_DIR,
    EMOTIONS,
    EVAL,
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
    TRAIN,
)
from .data import IndexedSubset, load_split
from .inference import load_model
from .plots import (
    plot_augmentations,
    plot_confusion_matrix,
    plot_dataset_grid,
    plot_gradcam,
    plot_per_class,
    plot_predictions,
    plot_training_curves,
    save_emotion_crops,
)


@torch.no_grad()
def predict_all(model, loader, device, tta: bool = True, tta_mode: str = "flip") -> np.ndarray:
    model.eval()
    all_probs = []
    for x, _, _ in loader:
        x = x.to(device)
        if device.type == "cpu":
            x = x.contiguous(memory_format=torch.channels_last)
        views = [x]
        if tta:
            views.append(torch.flip(x, dims=[3]))
        if tta and tta_mode == "multi":
            for size in (44, 52):
                scaled = torch.nn.functional.interpolate(
                    x, size=(size, size), mode="bilinear", align_corners=False, antialias=True
                )
                scaled = torch.nn.functional.interpolate(
                    scaled, size=x.shape[-2:], mode="bilinear", align_corners=False, antialias=True
                )
                views.append(scaled)
                views.append(torch.flip(scaled, dims=[3]))
        probs = torch.stack([torch.softmax(model(view), dim=1) for view in views]).mean(dim=0)
        all_probs.append(probs.float().cpu())
    return torch.cat(all_probs).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(EVAL.checkpoint))
    parser.add_argument("--label-set", choices=["fer2013", "ferplus"], default=EVAL.label_set)
    parser.add_argument("--batch-size", type=int, default=EVAL.batch_size)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--tta-mode", choices=["flip", "multi"], default="flip")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint}: epoch {ckpt.get('epoch', '?')}, val acc {ckpt.get('best_val_acc', 0):.4f}")

    images, labels = load_split("test", label_set=args.label_set)
    print(f"Test images: {len(images):,}")
    ds = IndexedSubset(images, labels)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    probs = predict_all(model, loader, device, tta=not args.no_tta, tta_mode=args.tta_mode)
    preds = probs.argmax(1)
    accuracy = float((preds == labels).mean())
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))
    cm = confusion_matrix(labels, preds, labels=list(range(len(EMOTIONS))))
    report = classification_report(
        labels, preds, labels=list(range(len(EMOTIONS))), target_names=EMOTIONS, output_dict=True, zero_division=0
    )

    support = {name: int((labels == i).sum()) for i, name in enumerate(EMOTIONS)}
    per_class = {
        name: {
            "precision": float(report[name]["precision"]),
            "recall": float(report[name]["recall"]),
            "f1": float(report[name]["f1-score"]),
            "support": support[name],
        }
        for name in EMOTIONS
    }

    off_diag = []
    row_sums = cm.sum(axis=1)
    for i, true_name in enumerate(EMOTIONS):
        for j, pred_name in enumerate(EMOTIONS):
            if i != j and cm[i, j] > 0:
                off_diag.append(
                    {
                        "true": true_name,
                        "predicted": pred_name,
                        "count": int(cm[i, j]),
                        "rate_of_true_class": float(cm[i, j] / max(1, row_sums[i])),
                    }
                )
    off_diag.sort(key=lambda d: d["count"], reverse=True)

    # Single-face CPU latency (the real-time path, no TTA).
    sample = torch.from_numpy(images[:32]).float().div(255).sub(0.5).div(0.5).unsqueeze(1)
    timings = []
    with torch.no_grad():
        for i in range(40):
            x = sample[i % 32].unsqueeze(0).to(device)
            if device.type == "cpu":
                x = x.contiguous(memory_format=torch.channels_last)
            start = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - start) * 1000)
    latency_ms = float(np.median(timings[5:]))

    metrics = {
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "test_weighted_f1": weighted_f1,
        "n_test": int(len(labels)),
        "label_set": args.label_set,
        "tta": (f"{args.tta_mode}" if not args.no_tta else False),
        "latency_ms": latency_ms,
        "checkpoint": str(args.checkpoint),
        "best_val_acc": float(ckpt.get("best_val_acc", 0.0)),
        "best_val_macro_f1": float(ckpt.get("best_val_macro_f1", 0.0)),
        "best_epoch": int(ckpt.get("epoch", 0)),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "top_confusions": off_diag[:8],
        "classes": EMOTIONS,
    }
    with open(EVAL.metrics, "w") as fh:
        json.dump(metrics, fh, indent=2)
    np.save(REPORT_DIR / "test_probs.npy", probs)
    print(f"Test accuracy: {accuracy * 100:.2f}% | macro F1: {macro_f1 * 100:.2f}%")
    print(f"Saved {EVAL.metrics} and {REPORT_DIR / 'test_probs.npy'}")
    print("\nPer class:")
    for name in EMOTIONS:
        row = per_class[name]
        print(
            f"  {name:9s} P {row['precision'] * 100:5.1f}  R {row['recall'] * 100:5.1f}  "
            f"F1 {row['f1'] * 100:5.1f}  n={row['support']}"
        )
    print("\nTop confusions:")
    for row in off_diag[:5]:
        print(
            f"  {row['true']:9s} -> {row['predicted']:9s} {row['count']:4d} "
            f"({row['rate_of_true_class'] * 100:.0f}% of {row['true']})"
        )

    if args.no_figures:
        return

    print("\nGenerating figures ...")
    history_path = TRAIN.history
    if Path(history_path).exists():
        with open(history_path) as fh:
            hist = json.load(fh)
        plot_training_curves(
            hist["history"],
            hist.get("best_epoch", metrics["best_epoch"]),
            FIGURE_DIR / "training_curves.png",
        )
        print("  training_curves.png")

    plot_confusion_matrix(cm, FIGURE_DIR / "confusion_matrix.png")
    print("  confusion_matrix.png")
    plot_per_class(report, support, FIGURE_DIR / "per_class_metrics.png")
    print("  per_class_metrics.png")
    plot_predictions(
        images, labels, probs, FIGURE_DIR / "predictions_grid.png", accuracy=accuracy
    )
    print("  predictions_grid.png")
    plot_gradcam(
        model,
        images,
        labels,
        probs,
        device,
        FIGURE_DIR / "gradcam_grid.png",
    )
    print("  gradcam_grid.png")

    tr_images, tr_labels = load_split("train", label_set=args.label_set)
    all_images = np.concatenate([tr_images, images], axis=0)
    all_labels = np.concatenate([tr_labels, labels], axis=0)
    plot_dataset_grid(all_images, all_labels, FIGURE_DIR / "dataset_grid.png")
    print("  dataset_grid.png")
    plot_dataset_grid(
        all_images, all_labels, FIGURE_DIR / "dataset_grid_clean.png", with_title=False
    )
    print("  dataset_grid_clean.png")
    plot_augmentations(all_images, all_labels, FIGURE_DIR / "augment_grid.png")
    save_emotion_crops(images, labels, ASSET_DIR / "faces")
    print(f"  face crops -> {ASSET_DIR / 'faces'}")

    print("Done.")


if __name__ == "__main__":
    main()
