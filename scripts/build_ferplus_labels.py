"""Derive FER+ consensus labels for the FER2013 images used in this project.

FER+ (Barsoum et al., 2016) re-annotated every FER2013 image with ten crowd
votes.  The pixels are identical; only the labels change.  This script maps
each locally stored image to its row in the original ``fer2013.csv`` by
nearest-neighbour on a 16x16 thumbnail, checks the match against the original
FER2013 label, and copies the FER+ majority label.

Output: ``data/raw/ferplus_labels.npz`` with ``train_labels`` / ``test_labels``
aligned to ``train.pt`` / ``test.pt``.  A label of -1 means the FER+ majority
was ``contempt`` (outside the seven-class protocol) and the image is dropped.

Usage:
    python scripts/build_ferplus_labels.py
    python scripts/build_ferplus_labels.py --fer2013-csv /tmp/fer2013.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import pickle
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import EMOTIONS, RAW_DIR  # noqa: E402

FERPLUS_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
FER2013_ZIP_URL = (
    "https://huggingface.co/datasets/chitradrishti/fer2013/resolve/main/fer2013.csv.zip"
)
FERPLUS_CLASSES = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]
# FER+ spelling -> the class order used throughout this project.
TO_PROJECT = {
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprise",
    "neutral": "neutral",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    print(f"downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def ensure_fer2013_csv(path: Path) -> None:
    if path.exists():
        return
    archive = path.with_suffix(".csv.zip")
    download(FER2013_ZIP_URL, archive)
    print(f"extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if name.endswith(".csv"))
        with zf.open(member) as src, open(path, "wb") as dst:
            dst.write(src.read())


def read_fer2013(path: Path) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Return thumbnails, FER2013 labels and usage strings in original row order."""
    thumbs: list[np.ndarray] = []
    labels: list[int] = []
    usage: list[str] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            image = np.fromstring(row[1], sep=" ", dtype=np.uint8).reshape(48, 48)
            thumbs.append(
                cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
                .astype(np.float32)
                .ravel()
                / 255.0
            )
            labels.append(int(row[0]))
            usage.append(row[2])
    return thumbs, labels, usage


def read_ferplus(path: Path) -> tuple[list[int], list[str]]:
    """Return majority label in project class order (-1 for contempt) and usage."""
    labels: list[int] = []
    usage: list[str] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            votes = [int(row[name]) for name in FERPLUS_CLASSES]
            winner = FERPLUS_CLASSES[int(np.argmax(votes))]
            if winner == "contempt":
                labels.append(-1)
            else:
                labels.append(EMOTIONS.index(TO_PROJECT[winner]))
            usage.append(row["Usage"])
    return labels, usage


def nearest_rows(
    images: list[np.ndarray], reference: np.ndarray, chunk: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Index of and squared distance to the closest reference thumbnail."""
    index: list[np.ndarray] = []
    distance: list[np.ndarray] = []
    ref_sq = (reference * reference).sum(1)
    for start in range(0, len(images), chunk):
        block = np.stack(images[start : start + chunk])
        d = (
            (block * block).sum(1)[:, None]
            + ref_sq[None, :]
            - 2.0 * block @ reference.T
        )
        best = d.argmin(1)
        index.append(best)
        distance.append(d[np.arange(len(block)), best])
    return np.concatenate(index), np.concatenate(distance)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fer2013-csv", type=Path, default=RAW_DIR / "fer2013.csv")
    parser.add_argument("--ferplus-csv", type=Path, default=RAW_DIR / "fer2013new.csv")
    parser.add_argument("--out", type=Path, default=RAW_DIR / "ferplus_labels.npz")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    args = parser.parse_args()

    ensure_fer2013_csv(args.fer2013_csv)
    download(FERPLUS_URL, args.ferplus_csv)

    print("reading FER+ annotations ...")
    plus_labels, plus_usage = read_ferplus(args.ferplus_csv)
    print("reading original FER2013 pixels ...")
    reference, base_labels, base_usage = read_fer2013(args.fer2013_csv)
    reference = np.stack(reference)

    if plus_usage != base_usage:
        raise SystemExit("FER+ and FER2013 row orders differ; refusing to guess.")
    print(f"row order verified for {len(plus_labels):,} images")

    out: dict[str, np.ndarray] = {}
    for split in ("train", "test"):
        rows = pickle.load(open(args.raw_dir / f"{split}.pt", "rb"))
        thumbs = [
            cv2.resize(
                cv2.imdecode(np.frombuffer(row["img_bytes"], np.uint8), cv2.IMREAD_GRAYSCALE),
                (16, 16),
                interpolation=cv2.INTER_AREA,
            )
            .astype(np.float32)
            .ravel()
            / 255.0
            for row in rows
        ]
        match, distance = nearest_rows(thumbs, reference)

        original = np.array([EMOTIONS.index(row["labels"]) for row in rows])
        matched_fer2013 = np.array([base_labels[i] for i in match])
        match_accuracy = float((matched_fer2013 == original).mean())
        mapped = np.array([plus_labels[i] if plus_labels[i] >= 0 else -1 for i in match])
        plus_agreement = float((mapped[mapped >= 0] == original[mapped >= 0]).mean())

        print(
            f"{split}: {len(rows):,} images | median thumbnail distance "
            f"{np.median(distance):.4f} | image match {match_accuracy:.2%} | "
            f"FER+ vs FER2013 labels {plus_agreement:.2%}"
        )
        if match_accuracy < 0.97 or np.median(distance) > 0.01:
            raise SystemExit("image matching looks wrong; refusing to write labels.")

        out[f"{split}_labels"] = mapped.astype(np.int64)
        out[f"{split}_source_row"] = match.astype(np.int32)

    out["classes"] = np.array(EMOTIONS)
    np.savez_compressed(args.out, **out)
    print(f"wrote {args.out}")
    for split in ("train", "test"):
        labels = out[f"{split}_labels"]
        kept = labels[labels >= 0]
        counts = np.bincount(kept, minlength=len(EMOTIONS))
        print(f"  {split}: {len(kept):,} kept, {int((labels < 0).sum()):,} contempt dropped")
        print("    " + " ".join(f"{name} {n}" for name, n in zip(EMOTIONS, counts)))


if __name__ == "__main__":
    main()
