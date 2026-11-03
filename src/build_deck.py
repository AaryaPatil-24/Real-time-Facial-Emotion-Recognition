"""Build presentation/index.html from the template + measured results."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from .config import ASSET_DIR, FIGURE_DIR, MODEL_DIR, PRESENTATION_DIR, REPORT_DIR, ROOT, TRAIN
from .inference import load_model
from .model import count_parameters

FIGURES = [
    "dataset_grid_clean.png",
    "augment_grid.png",
    "training_curves.png",
    "confusion_matrix.png",
    "per_class_metrics.png",
    "predictions_grid.png",
    "gradcam_grid.png",
    "realtime_demo.gif",
    "realtime_still.png",
]


FONTS = [
    ("Inter", 400, "inter-400.woff2"),
    ("Inter", 500, "inter-500.woff2"),
    ("Inter", 600, "inter-600.woff2"),
    ("Inter", 700, "inter-700.woff2"),
    ("Inter", 800, "inter-800.woff2"),
    ("DejaVuSymbols", 400, "dejavu-symbols-400.woff2"),
    ("DejaVuSymbols", 700, "dejavu-symbols-700.woff2"),
]


def inline_fonts(html: str) -> str:
    """Embed the woff2 files so the deck also works when opened from disk.

    The template registers them through the FontFace API from ArrayBuffers,
    because Chromium rejects these files as data-URI @font-face sources.
    """
    import base64
    import json

    payload = []
    for family, weight, name in FONTS:
        path = ASSET_DIR / "fonts" / name
        if not path.exists():
            print(f"  warning: missing font {path}")
            continue
        payload.append(
            {
                "family": family,
                "weight": weight,
                "b64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    marker = "window.__FONT_DATA__ = [];"
    if marker not in html:
        raise SystemExit("font placeholder missing from template")
    return html.replace(marker, "window.__FONT_DATA__ = " + json.dumps(payload) + ";", 1)


def main() -> None:
    metrics_path = REPORT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit("reports/metrics.json not found — run `python -m src.evaluate` first.")
    metrics = json.loads(metrics_path.read_text())
    history_path = TRAIN.history
    history = json.loads(history_path.read_text())["history"] if history_path.exists() else []

    acc = metrics["test_accuracy"] * 100
    f1 = metrics["test_macro_f1"] * 100
    ntest = metrics["n_test"]
    cm = metrics["confusion_matrix"]
    correct = int(sum(cm[i][i] for i in range(len(cm))))
    per_class = metrics["per_class"]
    top = metrics["top_confusions"][0] if metrics["top_confusions"] else None

    ckpt_path = Path(metrics.get("checkpoint", MODEL_DIR / "emotion_cnn_ferplus.pt"))
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    params = count_parameters(load_model(ckpt_path)[0])
    train_min = int(round(sum(h["seconds"] for h in history) / 60)) if history else 0
    epochs_run = len(history) if history else metrics.get("best_epoch", 40)
    latency = metrics.get("latency_ms", 0.0)
    fps_file = REPORT_DIR / "fps.txt"
    fps = fps_file.read_text().strip() if fps_file.exists() else "≈25"

    last = history[-1] if history else {"train_acc": 0.0, "val_acc": 0.0}
    gen_gap = abs(last["train_acc"] - last["val_acc"]) * 100
    macro_gap = acc - f1
    ferplus_baseline = 83.85  # Barsoum et al. (2016), VGG13 + majority voting
    acc_vs_bench = acc - ferplus_baseline
    top_true = top["true"] if top else "sad"
    top_pred = top["predicted"] if top else "neutral"
    top_rate = top["rate_of_true_class"] * 100 if top else 0.0

    values = {
        "ACC": f"{acc:.1f}%",
        "ACC_PCT": f"{acc:.1f}",
        "ACC_VS_BENCH": f"{acc_vs_bench:+.1f} pts",
        "BENCH": f"{ferplus_baseline:.2f}%",
        "F1": f"{f1:.1f}%",
        "VAL": f"{metrics.get('best_val_acc', 0) * 100:.1f}%",
        "CORRECT": f"{correct:,}",
        "NTEST": f"{ntest:,}",
        "PARAMS": f"{params:,}",
        "PARAMS_M": f"{params / 1e6:.2f}M",
        "TRAIN_MIN": str(train_min),
        "EPOCHS_RUN": f"{epochs_run} epochs",
        "BEST_EPOCH": str(metrics.get("best_epoch", "?")),
        "LATENCY": f"{latency:.1f} ms",
        "FPS": fps,
        "GEN_GAP": f"{gen_gap:.1f}",
        "MACRO_GAP": f"{macro_gap:.1f}",
        "TOP_CONF": f"{top_true} → {top_pred}",
        "TOP_CONF_TRUE": top_true,
        "TOP_CONF_PRED": top_pred,
        "TOP_CONF_RATE": f"{top_rate:.0f}%",
        "HAPPY_RECALL": f"{per_class['happy']['recall'] * 100:.0f}%",
        "DISGUST_RECALL": f"{per_class['disgust']['recall'] * 100:.0f}%",
        "TTA": "multi-scale TTA" if metrics.get("tta") == "multi" else "flip TTA",
        "TTA_SHORT": "multi-scale" if metrics.get("tta") == "multi" else "flip",
        "DATASET": "FER+" if metrics.get("label_set") == "ferplus" else "FER2013",
        "YEAR": "2026",
    }

    template = (PRESENTATION_DIR / "template.html").read_text()
    html = template
    for key, value in values.items():
        html = html.replace("{{" + key + "}}", value)
    leftovers = sorted(set(re.findall(r"\{\{[A-Z0-9_]+\}\}", html)))
    if leftovers:
        raise SystemExit(f"Unresolved template tokens: {leftovers}")
    html = inline_fonts(html)

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        src = FIGURE_DIR / name
        if not src.exists() and name == "dataset_grid_clean.png":
            src = FIGURE_DIR / "dataset_grid.png"
        if not src.exists():
            print(f"  warning: missing {src}")
            continue
        shutil.copy2(src, ASSET_DIR / name)

    out = PRESENTATION_DIR / "index.html"
    out.write_text(html)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  accuracy {acc:.2f}% | macro F1 {f1:.2f}% | latency {latency:.1f} ms | train {train_min} min")


if __name__ == "__main__":
    main()
