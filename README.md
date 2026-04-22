# karaoke-clipper

A pragmatic MVP that auto-detects likely sung-song segments from a YouTube VOD or local MP4, clips them, and writes reviewable manifests.

## Core Features
- Input modes:
  - YouTube URL (download via `yt-dlp`)
  - Local MP4 file
- Pipeline output:
  - master downloaded video for URL mode
  - extracted working WAV
  - auto-detected MP4 clips
  - optional WAV clips
  - `manifest.json` and `manifest.csv`
  - run logs
- API-free-first architecture:
  - local Chromaprint matching works without paid APIs
  - optional AcoustID backend is isolated behind an interface

## Architecture
Default flow:
1. Ingest source (`yt-dlp` URL mode or local file mode)
2. Extract mono working WAV with FFmpeg
3. Detect music-like candidate regions (`inaSpeechSegmenter`, fallback energy segmentation)
4. Match segments using local Chromaprint library
5. Optional AcoustID lookup if enabled and key is available
6. Refine boundaries with WhisperX when installed
7. Cut clips with FFmpeg and write manifest

## Repository Layout
- `app/main.py` - CLI + orchestration
- `app/config.py` - CLI/runtime config
- `app/ingest/youtube.py` - URL download + local source registration
- `app/preprocess/extract_audio.py` - working audio extraction
- `app/segment/music_segments.py` - segmentation + merge logic
- `app/identify/chromaprint_match.py` - local fingerprint matching
- `app/identify/acoustid_client.py` - optional AcoustID backend
- `app/align/whisperx_align.py` - timestamp refinement
- `app/clip/cutter.py` - clip export
- `app/output/manifest.py` - manifest writers
- `scripts/build_reference_library.py` - build local fingerprint index
- `tests/` - unit tests

## Prerequisites
- Python 3.10+
- FFmpeg (`ffmpeg`, `ffprobe`) on PATH
- Chromaprint tooling (`fpcalc`) for local matching

## Install
1. Create and install base environment:
   - `make setup`
2. Optional full ML stack for better segmentation/refinement:
   - `.venv/Scripts/python -m pip install -r requirements-ml.txt` (Windows)
   - `.venv/bin/python -m pip install -r requirements-ml.txt` (Unix)

## Build Local Fingerprint Library
Create an index from known songs folder:

`python scripts/build_reference_library.py --input-dir data/reference_songs --output data/reference_library.json`

Filename convention supported:
- `Artist - Song.wav`

## Run Examples
URL mode:

`python -m app.main --url "https://www.youtube.com/watch?v=<video_id>" --outdir output --audio-clips true --use-acoustid false --ref-library data/reference_library.json --device cpu`

Local file mode:

`python -m app.main --file "C:/videos/sample.mp4" --outdir output --audio-clips false --use-acoustid false --ref-library data/reference_library.json --device cpu`

Makefile shortcuts:
- `make run-url URL="https://www.youtube.com/watch?v=<video_id>"`
- `make run-file FILE="C:/videos/sample.mp4"`

Batch mode:

`python scripts/batch_run.py --input data/batch_sources.txt --outdir output/batch --ref-library data/reference_library.json`

## CLI Reference
- `--url <youtube_url>`
- `--file <local_mp4>`
- `--outdir <path>`
- `--audio-clips true|false`
- `--min-segment <seconds>`
- `--max-segment <seconds>`
- `--use-acoustid true|false`
- `--ref-library <path>`
- `--device cpu|cuda`

## Output Layout
```text
output/
  vods/
  audio/
  clips/
  manifests/
  logs/
  tmp/
```

Manifest fields include:
- `source_video`
- `video_id`
- `song`
- `artist`
- `start_sec`
- `end_sec`
- `start_tc`
- `end_tc`
- `confidence`
- `clip_path`
- `audio_path`
- `backend`

## Tests
Run unit tests:

`make test`

Covers:
- timecode conversion
- manifest writer
- segment merge logic

## Docker (CPU)
Build:

`make docker-build`

Run container:

`make docker-run`

## What Is Fully Working vs Fallback
Fully working in this MVP:
- URL/local ingest
- working audio extraction
- candidate segmentation (with fallback if `inaSpeechSegmenter` missing)
- local fingerprint matching (requires `pyacoustid` + `fpcalc`)
- clip export + manifest writing
- logging + retry-safe URL download

Fallback/stubbed behavior with clear TODOs:
- if `inaSpeechSegmenter` is not installed, fallback energy segmentation is used
- if `whisperx` is not installed, refinement returns coarse boundaries
- if `--use-acoustid true` but no key/package, backend is skipped safely

## Limitations
- Fingerprint matching quality depends on reference library coverage
- Live streams with heavy speech-over-music reduce detection quality
- WhisperX refinement quality depends on model availability and compute

## Future Improvements
- add stronger music/noise classifier and score calibration
- add confidence fusion across segmentation + fingerprint + transcript
- add richer report UI for human review
