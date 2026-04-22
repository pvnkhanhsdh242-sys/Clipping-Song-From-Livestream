VENV_DIR ?= .venv
PYTHON ?= python
URL ?=
FILE ?=
OUTDIR ?= output
AUDIO_CLIPS ?= false
MIN_SEGMENT ?= 8
MAX_SEGMENT ?= 240
USE_ACOUSTID ?= false
REF_LIBRARY ?= data/reference_library.json
DEVICE ?= cpu

ifeq ($(OS),Windows_NT)
	VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
else
	VENV_PYTHON := $(VENV_DIR)/bin/python
endif

.PHONY: setup run-url run-file test docker-build docker-run

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "Base setup complete. Optional: install -r requirements-ml.txt for inaSpeechSegmenter and WhisperX."

run-url:
	$(VENV_PYTHON) -m app.main --url "$(URL)" --outdir "$(OUTDIR)" --audio-clips "$(AUDIO_CLIPS)" --min-segment "$(MIN_SEGMENT)" --max-segment "$(MAX_SEGMENT)" --use-acoustid "$(USE_ACOUSTID)" --ref-library "$(REF_LIBRARY)" --device "$(DEVICE)"

run-file:
	$(VENV_PYTHON) -m app.main --file "$(FILE)" --outdir "$(OUTDIR)" --audio-clips "$(AUDIO_CLIPS)" --min-segment "$(MIN_SEGMENT)" --max-segment "$(MAX_SEGMENT)" --use-acoustid "$(USE_ACOUSTID)" --ref-library "$(REF_LIBRARY)" --device "$(DEVICE)"

test:
	$(VENV_PYTHON) -m pytest -q

docker-build:
	docker build -t karaoke-clipper:latest .

docker-run:
	docker run --rm -it -v "${PWD}/output:/app/output" -v "${PWD}/data:/app/data" karaoke-clipper:latest
