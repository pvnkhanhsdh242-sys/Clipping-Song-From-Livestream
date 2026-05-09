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

.PHONY: setup run-url run-file test docker-build docker-run docker-build-gpu docker-run-gpu docker-compose-gpu docker-streamlit-gpu docker-rebuild-gpu docker-healthcheck-gpu docker-clean-gpu docker-reset-gpu

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install --upgrade pip
	$(VENV_PYTHON) -m pip install -r requirements.txt
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
	docker build -f Dockerfile.gpu -t karaoke-clipper:gpu .

docker-run-gpu:
	docker run --rm -it --gpus all -v "${PWD}/output:/app/output" -v "${PWD}/data:/app/data" -v "${PWD}/secret:/app/secret" karaoke-clipper:gpu python scripts/container_runtime.py pipeline -- $(RUN_ARGS)

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
	-docker image rm -f karaoke-clipper:gpu random_project-karaoke-clipper-gpu random_project-karaoke-clipper-streamlit-gpu
	docker image prune -f

docker-reset-gpu:
	docker compose -f docker-compose.gpu.yml down --remove-orphans
	-docker image rm -f karaoke-clipper:gpu random_project-karaoke-clipper-gpu random_project-karaoke-clipper-streamlit-gpu
	docker image prune -f
	docker compose -f docker-compose.gpu.yml build karaoke-clipper-streamlit-gpu
	docker compose -f docker-compose.gpu.yml up karaoke-clipper-streamlit-gpu
