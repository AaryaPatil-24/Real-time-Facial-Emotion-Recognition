"""Benchmark the real-time loop headlessly and record the measured FPS.

Runs ``src.realtime`` on the simulated webcam source, parses its final average-FPS
line, and writes ``reports/fps.txt`` (used by the deck) plus a small JSON record.

Usage:
    python -m src.benchmark
    python -m src.benchmark --frames 300 --threads 4
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .config import REPORT_DIR, ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", type=int, default=240, help="frames to process after startup")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--checkpoint", default="models/emotion_cnn_ferplus.pt")
    parser.add_argument("--out", default=str(REPORT_DIR / "fps.txt"))
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.realtime",
        "--source",
        "demo",
        "--no-display",
        "--max-frames",
        str(args.frames),
        "--threads",
        str(args.threads),
        "--width",
        str(args.width),
        "--checkpoint",
        args.checkpoint,
    ]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"realtime benchmark failed with exit code {proc.returncode}")

    match = re.search(r"\((\d+(?:\.\d+)?) FPS average\)", proc.stdout)
    if not match:
        raise SystemExit("Could not find the FPS summary in the realtime output.")
    fps = float(match.group(1))
    text = f"{fps:.1f} FPS"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    record = {
        "fps": round(fps, 2),
        "frames": args.frames,
        "width": args.width,
        "threads": args.threads,
        "tta": True,
        "smoothing": 7,
        "source": "demo (held-out FER+ test faces)",
        "command": " ".join(cmd[1:]),
    }
    (REPORT_DIR / "benchmark.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {out}: {text}")
    print(f"Wrote {REPORT_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
