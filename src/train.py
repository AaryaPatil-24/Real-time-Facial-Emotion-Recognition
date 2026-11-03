"""Train the facial-expression CNN on FER+ (or FER2013) labels.

Usage:
    python -m src.train --epochs 40
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, replace

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .config import EMOTIONS, MODEL_DIR, REPORT_DIR, TRAIN
from .data import load_split, make_loaders, stratified_split
from .model import build_model, count_parameters


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ModelEMA:
    """Exponential moving average of model weights for a steadier checkpoint."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.module.parameters(), model.parameters()):
            ema_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for ema_buf, buf in zip(self.module.buffers(), model.buffers()):
            ema_buf.copy_(buf)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if device.type == "cpu":
            x = x.contiguous(memory_format=torch.channels_last)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        seen += len(y)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return (
        total_loss / seen,
        correct / seen,
        f1_score(labels, preds, average="macro"),
        preds,
        labels,
    )


def class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = (counts.sum() / (num_classes * np.maximum(counts, 1))) ** 0.5
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=TRAIN.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    parser.add_argument("--lr", type=float, default=TRAIN.lr)
    parser.add_argument("--seed", type=int, default=TRAIN.seed)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=TRAIN.patience)
    parser.add_argument("--arch", choices=["cnn", "resnet"], default=TRAIN.arch)
    parser.add_argument("--label-set", choices=["fer2013", "ferplus"], default=TRAIN.label_set)
    parser.add_argument("--width", type=int, default=TRAIN.width)
    parser.add_argument("--dropout", type=float, default=TRAIN.dropout)
    parser.add_argument("--val-fraction", type=float, default=TRAIN.val_fraction)
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=str(TRAIN.checkpoint))
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--history", type=str, default=None, help="Path to history json to resume")
    args = parser.parse_args()

    cfg = replace(
        TRAIN,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        num_workers=args.workers,
        patience=args.patience,
        val_fraction=args.val_fraction,
        dropout=args.dropout,
        width=args.width,
        augment=not args.no_augment,
        label_set=args.label_set,
    )

    set_seed(cfg.seed)
    threads = args.threads
    torch.set_num_threads(threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels_last = device.type == "cpu"
    print(f"Device: {device} | threads: {threads} | channels_last: {channels_last}")

    print(f"Loading {args.label_set} labels ...")
    images, labels = load_split("train", label_set=args.label_set)
    train_idx, val_idx = stratified_split(labels, cfg.val_fraction, cfg.seed)
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Classes: {EMOTIONS}")
    print("Class counts (train):", np.bincount(labels[train_idx], minlength=7).tolist())

    train_loader, val_loader = make_loaders(
        images[train_idx],
        labels[train_idx],
        images[val_idx],
        labels[val_idx],
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = build_model(
        arch=args.arch,
        num_classes=len(EMOTIONS),
        dropout=cfg.dropout,
        width=cfg.width,
    ).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    print(f"Architecture: {args.arch} (width {cfg.width}) | parameters: {count_parameters(model):,}")

    weights = None
    if cfg.use_class_weights and not args.no_class_weights:
        weights = class_weights(labels[train_idx]).to(device)
        print("Loss class weights:", [round(float(w), 3) for w in weights])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None

    warmup = min(3, max(1, cfg.epochs // 10))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, cfg.epochs - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_acc = 0.0
    best_epoch = -1
    patience_left = cfg.patience
    start_epoch = 1

    resume_path = args.resume
    if resume_path and os.path.exists(resume_path):
        try:
            ckpt = torch.load(resume_path, map_location=device)
            state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
            if state_dict is not None:
                try:
                    model.load_state_dict(state_dict)
                    print(f"Loaded weights from {resume_path}")
                except Exception as e:
                    print(f"Warning: failed to load weights: {e}")
            # try to load EMA if present
            if ema is not None and "ema_state_dict" in ckpt:
                try:
                    ema.module.load_state_dict(ckpt["ema_state_dict"])
                except Exception:
                    pass
            if "best_val_acc" in ckpt:
                best_acc = float(ckpt["best_val_acc"])
            if "epoch" in ckpt:
                start_epoch = int(ckpt["epoch"]) + 1
            print(f"Resuming from epoch {ckpt.get('epoch', '?')} -> starting at {start_epoch}")
        except Exception as e:
            print(f"Failed to resume from checkpoint: {e}")

    hist_path = args.history or str(cfg.history)
    if hist_path and os.path.exists(hist_path):
        try:
            with open(hist_path, "r") as fh:
                h = json.load(fh)
            if isinstance(h, dict) and "history" in h:
                history = h["history"]
                if h.get("best_val_acc") is not None and best_acc == 0.0:
                    best_acc = float(h["best_val_acc"])
                if h.get("best_epoch") is not None and best_epoch == -1:
                    best_epoch = int(h["best_epoch"])
                print(f"Loaded history from {hist_path} ({len(history)} epochs)")
        except Exception as e:
            print(f"Failed to load history: {e}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        started = time.time()
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs}", leave=False)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            if channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            bar.set_postfix(loss=f"{running_loss / seen:.3f}", acc=f"{correct / seen:.3f}")

        train_loss = running_loss / seen
        train_acc = correct / seen
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
        val_loss_ema = val_acc_ema = val_f1_ema = None
        use_ema = False
        if ema is not None:
            val_loss_ema, val_acc_ema, val_f1_ema, _, _ = evaluate(
                ema.module, val_loader, criterion, device
            )
            if (val_acc_ema, val_f1_ema) > (val_acc, val_f1):
                use_ema = True
        candidate_acc = val_acc_ema if use_ema else val_acc
        candidate_f1 = val_f1_ema if use_ema else val_f1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - started

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "val_macro_f1": round(val_f1, 5),
            "val_loss_ema": round(val_loss_ema, 5) if val_loss_ema is not None else None,
            "val_acc_ema": round(val_acc_ema, 5) if val_acc_ema is not None else None,
            "val_macro_f1_ema": round(val_f1_ema, 5) if val_f1_ema is not None else None,
            "lr": lr,
            "seconds": round(elapsed, 1),
        }
        history.append(row)
        ema_text = (
            f" | ema {val_loss_ema:.4f}/{val_acc_ema:.3f}"
            if val_acc_ema is not None
            else ""
        )
        print(
            f"epoch {epoch:02d} | train {train_loss:.4f}/{train_acc:.3f} | "
            f"val {val_loss:.4f}/{val_acc:.3f} | macro-F1 {val_f1:.3f}{ema_text} | "
            f"lr {lr:.1e} | {elapsed:.0f}s"
        )

        if candidate_acc > best_acc:
            best_acc, best_epoch = candidate_acc, epoch
            patience_left = cfg.patience
            save_model = ema.module if use_ema else model
            torch.save(
                {
                    "state_dict": save_model.state_dict(),
                    "classes": EMOTIONS,
                    "arch": args.arch,
                    "label_set": args.label_set,
                    "dropout": cfg.dropout,
                    "width": cfg.width,
                    "epoch": epoch,
                    "best_val_acc": best_acc,
                    "best_val_macro_f1": candidate_f1,
                    "ema": use_ema,
                    "config": {
                        "epochs": cfg.epochs,
                        "batch_size": cfg.batch_size,
                        "lr": cfg.lr,
                        "weight_decay": cfg.weight_decay,
                        "dropout": cfg.dropout,
                        "label_smoothing": cfg.label_smoothing,
                        "val_fraction": cfg.val_fraction,
                        "seed": cfg.seed,
                        "arch": args.arch,
                        "label_set": args.label_set,
                        "width": cfg.width,
                        "ema_decay": args.ema_decay,
                        "class_weights": weights is not None,
                    },
                },
                args.checkpoint,
            )
            print(f"  saved new best checkpoint -> {args.checkpoint} ({'ema' if use_ema else 'raw'})")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

        with open(hist_path, "w") as fh:
            json.dump(
                {
                    "history": history,
                    "best_val_acc": best_acc,
                    "best_epoch": best_epoch,
                    "config": asdict(cfg),
                    "checkpoint": args.checkpoint,
                    "ema_decay": args.ema_decay,
                    "threads": args.threads,
                },
                fh,
                indent=2,
                default=str,
            )

    with open(hist_path, "w") as fh:
        json.dump(
            {
                "history": history,
                "best_val_acc": best_acc,
                "best_epoch": best_epoch,
                "config": asdict(cfg),
                "checkpoint": args.checkpoint,
                "ema_decay": args.ema_decay,
                "threads": args.threads,
            },
            fh,
            indent=2,
            default=str,
        )
    print(f"Done. Best val accuracy {best_acc:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
