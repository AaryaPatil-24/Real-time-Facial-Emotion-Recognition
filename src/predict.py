"""Run the trained model on an image or a folder of images.

Examples:
    python -m src.predict --image photo.jpg --out reports/photo_annotated.jpg
    python -m src.predict --image samples/ --out reports/samples/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import EMOTIONS, EMOTION_COLORS
from .inference import FaceDetector, crop_face, load_model, predict_face


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def annotate(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    probs: np.ndarray,
    top_k: int = 3,
) -> np.ndarray:
    out = frame.copy()
    x, y, w, h = box
    order = np.argsort(probs)[::-1]
    label = EMOTIONS[order[0]]
    color = hex_to_bgr(EMOTION_COLORS[label])
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

    title = f"{label.upper()}  {probs[order[0]] * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
    cv2.rectangle(out, (x, max(0, y - th - 14)), (x + tw + 14, y), color, -1)
    cv2.putText(
        out,
        title,
        (x + 7, y - 8),
        cv2.FONT_HERSHEY_DUPLEX,
        0.75,
        (15, 23, 42),
        1,
        cv2.LINE_AA,
    )

    lines = [f"{EMOTIONS[i]:<9s} {probs[i] * 100:5.1f}%" for i in order[:top_k]]
    pad, line_h = 10, 24
    box_h = line_h * len(lines) + pad * 2
    box_w = 190
    yy = y + h + 10 if y + h + 10 + box_h < out.shape[0] else max(0, y - box_h - 10)
    cv2.rectangle(out, (x, yy), (min(out.shape[1], x + box_w), yy + box_h), (15, 23, 42), -1)
    for i, line in enumerate(lines):
        txt_color = (255, 255, 255) if i == 0 else (203, 213, 225)
        cv2.putText(
            out,
            line,
            (x + pad, yy + pad + 16 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            txt_color,
            1,
            cv2.LINE_AA,
        )
    return out


def run_image(path: Path, model, device, detector, out_path: Path, no_detect: bool):
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    if no_detect:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = [(0, 0, frame.shape[1], frame.shape[0])]
        crop, box = cv2.resize(gray, (48, 48)), faces[0]
    else:
        faces = detector.detect(frame)
        if not faces:
            print(f"  no face detected, using whole image: {path.name}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            crop, box = cv2.resize(gray, (48, 48)), (0, 0, frame.shape[1], frame.shape[0])
        else:
            crop, box = crop_face(frame, faces[0])

    probs = predict_face(model, crop, device=device)
    top = int(probs.argmax())
    print(f"{path.name}: {EMOTIONS[top]} ({probs[top] * 100:.1f}%) | " + ", ".join(
        f"{EMOTIONS[i]} {probs[i] * 100:.1f}%" for i in np.argsort(probs)[::-1][:3]
    ))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotate(frame, box, probs))
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Image file or directory")
    parser.add_argument("--out", default="reports/predictions", help="Output file or directory")
    parser.add_argument("--checkpoint", default="models/emotion_cnn_ferplus.pt")
    parser.add_argument("--no-detect", action="store_true", help="Skip face detection")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = load_model(args.checkpoint, device)
    detector = FaceDetector()
    src, out = Path(args.image), Path(args.out)

    if src.is_dir():
        out.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            p for p in src.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        for path in paths:
            run_image(path, model, device, detector, out / f"{path.stem}_annotated.jpg", args.no_detect)
    else:
        out_path = out if out.suffix else out / f"{src.stem}_annotated.jpg"
        run_image(src, model, device, detector, out_path, args.no_detect)


if __name__ == "__main__":
    main()
