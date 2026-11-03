"""Real-time facial emotion recognition from a webcam, video, image or demo source.

Examples:
    python -m src.realtime                        # default webcam
    python -m src.realtime --source 1             # second camera
    python -m src.realtime --source clip.mp4 --record reports/live.mp4
    python -m src.realtime --source demo          # no webcam required
    python -m src.realtime --source demo --no-display --max-frames 240 \
        --record reports/demo.mp4

Keys: Q/ESC quit · G toggle Grad-CAM · B toggle bars · S screenshot · SPACE pause
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from .config import EMOTIONS, EMOTION_COLORS, FIGURE_DIR, MODEL_DIR
from .data import load_split
from .gradcam import GradCAM, cam_overlay
from .inference import FaceDetector, crop_face, load_model, predict_face


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


COLORS_BGR = {name: hex_to_bgr(color) for name, color in EMOTION_COLORS.items()}
INK = (15, 23, 42)
WHITE = (241, 245, 249)
MUTED = (148, 163, 184)


class DemoSource:
    """Synthetic 'webcam' built from held-out FER+ test faces (for machines without a camera)."""

    def __init__(self, width: int = 960, height: int = 540, seed: int = 7):
        print("Loading held-out test faces for demo ...")
        self.images, self.labels = load_split("test")
        rng = np.random.default_rng(seed)
        self.order = rng.permutation(len(self.images))
        self.i = -1
        self.t = 0
        self.width, self.height = width, height
        self.face = None
        self._pick_next()
        xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
        yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
        base = 16 + 14 * yy + 10 * (1 - xx) ** 2
        self.bg = np.dstack([base * 0.75, base * 0.95, base * 1.25]).clip(0, 255).astype(np.uint8)

    def _pick_next(self):
        self.i = (self.i + 1) % len(self.order)
        idx = int(self.order[self.i])
        img = self.images[idx]
        if img.mean() < 40:  # skip near-black frames
            self.images[idx] = np.clip(img.astype(np.float32) + 20, 0, 255).astype(np.uint8)
        face = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
        self.face = cv2.cvtColor(face, cv2.COLOR_GRAY2BGR)
        self.t = 0

    def read(self):
        self.t += 1
        if self.t > 45:
            self._pick_next()
        frame = self.bg.copy()
        dx = int(8 * np.sin(self.t / 14))
        dy = int(5 * np.cos(self.t / 17))
        cx, cy = self.width // 2 + dx, self.height // 2 + dy - 10
        x0, y0 = cx - 150, cy - 150
        frame[y0 : y0 + 300, x0 : x0 + 300] = self.face
        cv2.putText(
            frame,
            "SIMULATED WEBCAM  ·  HELD-OUT TEST FACE",
            (16, self.height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            MUTED,
            1,
            cv2.LINE_AA,
        )
        return True, frame

    def release(self):
        pass

    def is_demo(self) -> bool:
        return True


class ImageSource:
    def __init__(self, path: Path, width: int = 960):
        frame = cv2.imread(str(path))
        if frame is None:
            raise SystemExit(f"Could not read image: {path}")
        h, w = frame.shape[:2]
        if w > width:
            frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
        self.frame = frame

    def read(self):
        return True, self.frame.copy()

    def release(self):
        pass

    def is_demo(self) -> bool:
        return False


def open_source(source: str, width: int = 960):
    if source == "demo":
        return DemoSource(width=width)
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        path = Path(source)
        if not path.exists():
            raise SystemExit(f"Source not found: {source}")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return ImageSource(path, width=width)
        cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(width * 9 / 16))
    return cap


def rounded_panel(frame, x0, y0, x1, y1, alpha=0.72):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), INK, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def draw_bars(frame, probs, x0, y0, width=250):
    order = np.argsort(probs)[::-1]
    row_h = 25
    rounded_panel(frame, x0 - 10, y0 - 12, x0 + width + 10, y0 + row_h * len(order) + 2)
    for rank, cls in enumerate(order):
        y = y0 + rank * row_h
        color = COLORS_BGR[EMOTIONS[cls]]
        cv2.putText(
            frame, EMOTIONS[cls][:9], (x0, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, WHITE, 1, cv2.LINE_AA
        )
        bar_x, bar_w = x0 + 78, width - 112
        cv2.rectangle(frame, (bar_x, y + 2), (bar_x + bar_w, y + 14), (51, 65, 85), -1)
        fill = int(bar_w * float(probs[cls]))
        cv2.rectangle(frame, (bar_x, y + 2), (bar_x + fill, y + 14), color, -1)
        cv2.putText(
            frame,
            f"{probs[cls] * 100:4.0f}%",
            (bar_x + bar_w + 8, y + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            MUTED,
            1,
            cv2.LINE_AA,
        )


def draw_face_box(frame, box, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2, cv2.LINE_AA)
    length = max(10, min(w, h) // 6)
    for cx, cy, dx, dy in [
        (x, y, 1, 1),
        (x + w, y, -1, 1),
        (x, y + h, 1, -1),
        (x + w, y + h, -1, -1),
    ]:
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, 4, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, 4, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="0", help="webcam index, video/image path, or 'demo'")
    parser.add_argument("--checkpoint", default=str(MODEL_DIR / "emotion_cnn_ferplus.pt"))
    parser.add_argument("--record", default=None, help="Write annotated video here (mp4/avi)")
    parser.add_argument("--no-display", action="store_true", help="Headless mode (use with --record)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)")
    parser.add_argument("--gradcam", action="store_true", help="Start with Grad-CAM enabled")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE to the face crop")
    parser.add_argument("--smooth", type=int, default=7, help="Temporal averaging window")
    parser.add_argument("--no-tta", action="store_true", help="Disable horizontal-flip TTA")
    parser.add_argument("--mirror", action="store_true", help="Mirror the camera preview")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')}, val acc {ckpt.get('best_val_acc', 0):.3f})")

    detector = FaceDetector(min_size=48 if args.source == "demo" else 56)
    cam = GradCAM(model, model.cam_target)
    source = open_source(args.source, width=args.width)
    is_camera = args.source.isdigit()
    is_demo = args.source == "demo"

    writer = None
    if args.record:
        rec_path = Path(args.record)
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if rec_path.suffix == ".avi" else "mp4v"))
        writer = cv2.VideoWriter(str(rec_path), fourcc, 25.0, (args.width, int(args.width * 9 / 16)))
        if not writer.isOpened():
            raise SystemExit(f"Could not open video writer: {rec_path}")

    history: deque[np.ndarray] = deque(maxlen=max(1, args.smooth))
    show_bars = True
    show_cam = args.gradcam
    paused = False
    last_frame = None
    fps = 0.0
    last_t = time.perf_counter()
    frames = 0
    pred = None
    box = None
    started = time.time()

    print("Running. Press Q/ESC to quit, G for Grad-CAM, B for bars, S for screenshot.")
    try:
        while True:
            if not paused:
                ok, frame = source.read()
                if not ok:
                    break
                if is_camera and args.mirror:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if w > args.width:
                    frame = cv2.resize(frame, (args.width, int(h * args.width / w)), interpolation=cv2.INTER_AREA)
                if frame.shape[1] != args.width or frame.shape[0] != int(args.width * 9 / 16):
                    frame = cv2.resize(frame, (args.width, int(args.width * 9 / 16)))

                faces = detector.detect(frame)
                now = time.perf_counter()
                inst_fps = 1.0 / max(1e-6, now - last_t)
                fps = inst_fps if fps == 0 else 0.9 * fps + 0.1 * inst_fps
                last_t = now
                frames += 1

                if faces:
                    box = faces[0]
                    gray48, expanded = crop_face(frame, box, margin=0.10)
                    box = expanded
                    probs = predict_face(
                        model, gray48, device=device, tta=not args.no_tta, use_clahe=args.clahe
                    )
                    history.append(probs)
                    probs = np.mean(history, axis=0)
                    pred = probs
                else:
                    pred = None
                    history.clear()

                display = frame.copy()
                if pred is not None and box is not None:
                    label = EMOTIONS[int(pred.argmax())]
                    color = COLORS_BGR[label]
                    if show_cam:
                        gray48, _ = crop_face(frame, box, margin=0.10)
                        cam_map, _ = cam(
                            torch.from_numpy(gray48).float().div(255).sub(0.5).div(0.5)[None, None].to(device),
                            class_idx=int(pred.argmax()),
                        )
                        x, y, wb, hb = box
                        overlay = cam_overlay(gray48, cam_map)
                        overlay = cv2.resize(overlay, (wb, hb), interpolation=cv2.INTER_LINEAR)
                        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                        roi = display[y : y + hb, x : x + wb]
                        if roi.shape == overlay.shape:
                            display[y : y + hb, x : x + wb] = cv2.addWeighted(roi, 0.35, overlay, 0.65, 0)
                    draw_face_box(display, box, color)

                    panel_w = 330
                    rounded_panel(display, 14, 14, 14 + panel_w, 88)
                    cv2.putText(
                        display,
                        label.upper(),
                        (30, 58),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.15,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        f"{pred.max() * 100:.1f}% confidence",
                        (30, 78),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        MUTED,
                        1,
                        cv2.LINE_AA,
                    )
                    if show_bars:
                        draw_bars(display, pred, 30, display.shape[0] - 202, width=270)
                else:
                    cv2.putText(
                        display,
                        "NO FACE DETECTED",
                        (30, 60),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.9,
                        (239, 68, 68),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.putText(
                    display,
                    f"{fps:4.1f} FPS",
                    (display.shape[1] - 110, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    MUTED,
                    1,
                    cv2.LINE_AA,
                )
                if show_cam:
                    cv2.putText(
                        display,
                        "GRAD-CAM",
                        (display.shape[1] - 110, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (245, 158, 11),
                        1,
                        cv2.LINE_AA,
                    )
                last_frame = display
                if writer is not None:
                    writer.write(display)

            if not args.no_display:
                cv2.imshow("Facial Emotion Recognition — CNN", last_frame if last_frame is not None else frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("g"):
                    show_cam = not show_cam
                elif key == ord("b"):
                    show_bars = not show_bars
                elif key == ord("s") and last_frame is not None:
                    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
                    out = FIGURE_DIR / "live_capture.jpg"
                    cv2.imwrite(str(out), last_frame)
                    print(f"Saved screenshot: {out}")
                elif key == ord(" "):
                    paused = not paused

            if args.max_frames and frames >= args.max_frames:
                break
            if args.no_display and not args.record and not is_demo and not is_camera:
                pass
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if writer is not None:
            writer.release()
            print(f"Saved video: {args.record}")
        if not args.no_display:
            cv2.destroyAllWindows()
        elapsed = max(1e-6, time.time() - started)
        print(f"Processed {frames} frames in {elapsed:.1f}s ({frames / elapsed:.1f} FPS average)")


if __name__ == "__main__":
    main()
