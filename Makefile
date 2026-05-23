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
CLIP_RESOLUTION ?= source
EXPECTED_SONG_COUNT ?=
RUN_ARGS ?= --help

EXPECTED_SONG_ARG :=
ifneq ($(strip $(EXPECTED_SONG_COUNT)),)
EXPECTED_SONG_ARG := --expected-song-count "$(EXPECTED_SONG_COUNT)"
endif

ifeq ($(OS),Windows_NT)
	VENV_PYTHON := $(VENV_DIR)/Scripts/python.exe
else
	VENV_PYTHON := $(VENV_DIR)/bin/python
endif

.PHONY: setup run-url run-file test docker-build docker-run docker-build-gpu docker-run-gpu docker-build-train-gpu docker-build-gpu-full-ml docker-train-gpu docker-compose-gpu docker-streamlit-gpu docker-rebuild-gpu docker-healthcheck-gpu docker-clean-gpu docker-reset-gpu

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements-dev.txt
	@echo "Base setup complete. Optional: install -r requirements-ml.txt for inaSpeechSegmenter and WhisperX."

run-url:
	$(VENV_PYTHON) -m app.main --url "$(URL)" --outdir "$(OUTDIR)" --audio-clips "$(AUDIO_CLIPS)" --min-segment "$(MIN_SEGMENT)" --max-segment "$(MAX_SEGMENT)" --use-acoustid "$(USE_ACOUSTID)" --ref-library "$(REF_LIBRARY)" --device "$(DEVICE)" --clip-resolution "$(CLIP_RESOLUTION)" $(EXPECTED_SONG_ARG)

run-file:
	$(VENV_PYTHON) -m app.main --file "$(FILE)" --outdir "$(OUTDIR)" --audio-clips "$(AUDIO_CLIPS)" --min-segment "$(MIN_SEGMENT)" --max-segment "$(MAX_SEGMENT)" --use-acoustid "$(USE_ACOUSTID)" --ref-library "$(REF_LIBRARY)" --device "$(DEVICE)" --clip-resolution "$(CLIP_RESOLUTION)" $(EXPECTED_SONG_ARG)

test:
	$(VENV_PYTHON) -m pytest -q

docker-build:
	docker build -t karaoke-clipper:latest .

docker-run:
	docker run --rm -it -v "${PWD}/output:/app/output" -v "${PWD}/data:/app/data" karaoke-clipper:latest

docker-build-gpu:
	docker build -f Dockerfile.gpu --target base-gpu -t karaoke-clipper:gpu -t karaoke-clipper:train-gpu .

docker-run-gpu:
	docker run --rm -it --gpus all -v "${PWD}/output:/app/output" -v "${PWD}/data:/app/data" -v "${PWD}/secret:/app/secret" karaoke-clipper:gpu python scripts/container_runtime.py pipeline -- $(RUN_ARGS)

docker-build-train-gpu:
	docker build -f Dockerfile.gpu --target base-gpu -t karaoke-clipper:gpu -t karaoke-clipper:train-gpu .

docker-build-gpu-full-ml:
	docker build -f Dockerfile.gpu --target full-ml -t karaoke-clipper:gpu-full-ml .

docker-train-gpu:
	docker run --rm --gpus all -v "${PWD}/output:/app/output" -v "${PWD}/data:/app/data" -v "${PWD}/secret:/app/secret" -v "${PWD}/.cache:/app/.cache" karaoke-clipper:train-gpu python scripts/container_runtime.py run --require-cuda -- python scripts/train_singing_model_all.py --backend pytorch --device cuda --epochs 5 $(RUN_ARGS)

docker-compose-gpu:
	docker compose -f docker-compose.gpu.yml up karaoke-clipper-streamlit-gpu

docker-streamlit-gpu:
	docker compose -f docker-compose.gpu.yml up karaoke-clipper-streamlit-gpu

docker-rebuild-gpu:
	docker compose -f docker-compose.gpu.yml build karaoke-clipper-streamlit-gpu

docker-healthcheck-gpu:
	docker run --rm --gpus all karaoke-clipper:gpu python scripts/container_runtime.py healthcheck

docker-clean-gpu:
	docker compose -f docker-compose.gpu.yml down --remove-orphans
	-docker image rm -f random_project-karaoke-clipper-gpu random_project-karaoke-clipper-streamlit-gpu pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
	docker image prune -f

docker-reset-gpu:
	docker compose -f docker-compose.gpu.yml down --remove-orphans
	-docker image rm -f random_project-karaoke-clipper-gpu random_project-karaoke-clipper-streamlit-gpu pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
	docker image prune -f
	docker compose -f docker-compose.gpu.yml build karaoke-clipper-streamlit-gpu
	docker compose -f docker-compose.gpu.yml up karaoke-clipper-streamlit-gpu
