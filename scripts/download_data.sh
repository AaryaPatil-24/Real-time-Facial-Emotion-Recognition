#!/usr/bin/env bash
# Download the FER2013 train/test splits (pickled JPEG bytes) from Hugging Face.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
base="https://huggingface.co/datasets/Jeneral/fer-2013/resolve/main"
for split in train test; do
  if [ ! -f "data/raw/${split}.pt" ]; then
    echo "Downloading ${split}.pt ..."
    curl -L --fail --retry 3 -o "data/raw/${split}.pt" "${base}/${split}.pt"
  else
    echo "data/raw/${split}.pt already exists, skipping."
  fi
done
ls -lh data/raw
