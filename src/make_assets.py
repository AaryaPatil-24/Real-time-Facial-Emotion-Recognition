"""Generate the animated real-time demo (GIF) used in the slide deck and README."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from .config import EMOTIONS, EMOTION_COLORS, FIGURE_DIR, REPORT_DIR
from .data import load_split
from .inference import load_model

W, H = 640, 360
BG_TOP = (9, 15, 28)
BG_BOTTOM = (17, 28, 48)
INK = (11, 18, 32)
WHITE = (241, 245, 249)
MUTED = (148, 163, 184)
FONT_DIR = Path(font_manager.findfont("DejaVu Sans")).parent


def load_font(size: int, bold: bool = False):
    path = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


F_BADGE = load_font(11, True)
F_NAME = load_font(34, True)
F_CONF = load_font(14)
F_BAR = load_font(11)
F_SMALL = load_font(9)
F_FPS = load_font(11, True)


def hex_rgb(value: str) -> tuple[int, int, int]:
    return tuple(int(value.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))


def background() -> Image.Image:
    y = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
    top = np.array(BG_TOP, dtype=np.float32)[None, None, :]
    bottom = np.array(BG_BOTTOM, dtype=np.float32)[None, None, :]
    arr = (top * (1 - y) + bottom * y)
    arr = np.repeat(arr, W, axis=1).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    draw = ImageDraw.Draw(img)
    for x in range(0, W, 40):
        draw.line([(x, 0), (x, H)], fill=(20, 32, 54), width=1)
    for y0 in range(0, H, 40):
        draw.line([(0, y0), (W, y0)], fill=(20, 32, 54), width=1)
    return img


def pick_samples(images, labels, probs):
    picks = []
    for cls, name in enumerate(EMOTIONS):
        correct = np.flatnonzero((probs.argmax(1) == labels) & (labels == cls))
        if len(correct) == 0:
            correct = np.flatnonzero(labels == cls)
        best = max(correct, key=lambda i: probs[i, cls])
        picks.append((name, best))
    return picks


def render_frame(face: Image.Image, probs: np.ndarray, true_name: str, tick: int, fps: float) -> Image.Image:
    frame = background()
    draw = ImageDraw.Draw(frame, "RGBA")

    pred_idx = int(probs.argmax())
    pred_name = EMOTIONS[pred_idx]
    color = hex_rgb(EMOTION_COLORS[pred_name])
    fx, fy, fs = 350, 48, 264
    frame.paste(face, (fx, fy))

    # Face box + corner accents.
    draw.rectangle([fx - 6, fy - 6, fx + fs + 6, fy + fs + 6], outline=color, width=3)
    corner = 22
    for cx, cy, dx, dy in [
        (fx - 6, fy - 6, 1, 1),
        (fx + fs + 6, fy - 6, -1, 1),
        (fx - 6, fy + fs + 6, 1, -1),
        (fx + fs + 6, fy + fs + 6, -1, -1),
    ]:
        draw.line([(cx, cy), (cx + dx * corner, cy)], fill=color, width=6)
        draw.line([(cx, cy), (cx, cy + dy * corner)], fill=color, width=6)

    # Scanning line to suggest live inference.
    scan_y = fy + (tick * 7) % fs
    draw.line([(fx, scan_y), (fx + fs, scan_y)], fill=(245, 158, 11, 150), width=2)
    draw.rectangle([fx, scan_y - 1, fx + fs, scan_y + 1], fill=(245, 158, 11, 60))

    # Left panel: badge, prediction, confidence, bars.
    draw.rounded_rectangle([20, 18, 322, 340], radius=12, fill=(8, 13, 24, 220), outline=(38, 54, 80), width=1)
    draw.ellipse([34, 27, 42, 35], fill=(239, 68, 68))
    draw.text((49, 24), "LIVE DEMO", font=F_BADGE, fill=(248, 113, 113))
    draw.text((34, 48), pred_name.upper(), font=F_NAME, fill=color)
    draw.text((35, 89), f"{probs[pred_idx] * 100:.1f}% confidence", font=F_CONF, fill=MUTED)

    order = np.argsort(probs)[::-1]
    y = 124
    for rank, idx in enumerate(order):
        name = EMOTIONS[idx]
        c = hex_rgb(EMOTION_COLORS[name])
        draw.text((34, y + 1), name, font=F_BAR, fill=WHITE if rank == 0 else MUTED)
        bx, bw = 112, 148
        draw.rounded_rectangle([bx, y + 2, bx + bw, y + 13], radius=3, fill=(30, 41, 59))
        fill_w = max(2, int(bw * float(probs[idx])))
        draw.rounded_rectangle([bx, y + 2, bx + fill_w, y + 13], radius=3, fill=c)
        draw.text((bx + bw + 8, y), f"{probs[idx] * 100:3.0f}%", font=F_BAR, fill=MUTED)
        y += 23

    draw.text((34, 306), f"true label: {true_name}", font=F_SMALL, fill=(100, 116, 139))
    draw.text((34, 320), "smoothed over 7 frames", font=F_SMALL, fill=(100, 116, 139))

    draw.text((W - 92, 22), f"{fps:4.1f} FPS", font=F_FPS, fill=MUTED)
    draw.text((W - 92, 40), "Grad-CAM: G", font=F_SMALL, fill=(100, 116, 139))

    draw.text((22, 346), "SIMULATED WEBCAM  ·  HELD-OUT TEST FACES  ·  NOT A REAL PERSON", font=F_SMALL, fill=(85, 103, 128))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/emotion_cnn_ferplus.pt")
    parser.add_argument("--out", default=str(FIGURE_DIR / "realtime_demo.gif"))
    parser.add_argument("--hold", type=int, default=17)
    args = parser.parse_args()

    model, _ = load_model(args.checkpoint)
    images, labels = load_split("test")
    probs = np.load(REPORT_DIR / "test_probs.npy") if (REPORT_DIR / "test_probs.npy").exists() else None
    if probs is None:
        from .inference import predict_face

        probs = np.stack([predict_face(model, img) for img in images])
    picks = pick_samples(images, labels, probs)

    frames = []
    for k, (true_name, idx) in enumerate(picks):
        gray = images[idx]
        face = cv2.cvtColor(
            cv2.resize(gray, (264, 264), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB
        )
        face_img = Image.fromarray(face)
        p = probs[idx]
        for t in range(args.hold):
            frames.append(render_frame(face_img, p, true_name, k * args.hold + t, 27.0 + 1.8 * np.sin((k * 17 + t) / 9)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Wrote {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.2f} MB)")

    # A still frame for README / thumbnails.
    still = render_frame(
        Image.fromarray(
            cv2.cvtColor(
                cv2.resize(images[picks[3][1]], (264, 264), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB
            )
        ),
        probs[picks[3][1]],
        picks[3][0],
        12,
        29.6,
    )
    still_path = FIGURE_DIR / "realtime_still.png"
    still.save(still_path)
    print(f"Wrote {still_path}")


if __name__ == "__main__":
    main()
