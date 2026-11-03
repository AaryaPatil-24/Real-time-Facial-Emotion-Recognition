PY := .venv/bin/python
PIP := .venv/bin/pip

.PHONY: help setup data train eval demo deck pdf clean

help:
	@echo "setup   create .venv and install dependencies"
	@echo "data    download FER2013 images and build FER+ labels"
	@echo "train   train the residual CNN (50 epochs, CPU)"
	@echo "eval    evaluate, write metrics + figures"
	@echo "demo    real-time webcam demo"
	@echo "deck    rebuild presentation/index.html"
	@echo "pdf     export presentation/deck.pdf (needs playwright)"

setup:
	python3 -m venv .venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

data:
	bash scripts/download_data.sh
	$(PY) scripts/build_ferplus_labels.py

train:
	$(PY) -m src.train --arch resnet --width 32 --label-set ferplus --epochs 40

eval:
	$(PY) -m src.evaluate --label-set ferplus --tta-mode multi
	$(PY) -m src.make_assets
	$(PY) -m src.benchmark --frames 240

demo:
	$(PY) -m src.realtime --source demo

deck:
	$(PY) -m src.build_deck

pdf:
	$(PY) -m src.export_deck

clean:
	rm -rf reports/figures reports/metrics.json reports/history.json \
		presentation/index.html presentation/deck.pdf presentation/slides
