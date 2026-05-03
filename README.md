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
  - Note: optional ML extras (for `inaSpeechSegmenter`/`whisperx`) typically require a Python version that supports TensorFlow wheels (recommended Python 3.9-3.12).

## Build Local Fingerprint Library
Create an index from known songs folder:

`python scripts/build_reference_library.py --input-dir data/reference_songs --output data/reference_library.json`

Filename convention supported:
- `Artist - Song.wav`

## Run Examples
URL mode:

`python -m app.main --url "https://www.youtube.com/watch?v=<video_id>" --outdir output --audio-clips true --use-acoustid false --ref-library data/reference_library.json --device cpu --clip-resolution 720p --expected-song-count 16`

Local file mode:

`python -m app.main --file "C:/videos/sample.mp4" --outdir output --audio-clips false --use-acoustid false --ref-library data/reference_library.json --device cpu --clip-resolution source`

Makefile shortcuts:
- `make run-url URL="https://www.youtube.com/watch?v=<video_id>"`
- `make run-file FILE="C:/videos/sample.mp4"`

Batch mode:

`python scripts/batch_run.py --input data/batch_sources.txt --outdir output/batch --ref-library data/reference_library.json`

Batch mode with segment tuning:

`python scripts/batch_run.py --input data/batch_sources.txt --outdir output/batch --min-segment 60 --max-segment 420 --merge-gap 3.5 --expected-song-count 16`

## Streamlit UI
Run a local UI to enter parameters, preview timestamps, and launch the pipeline:

`streamlit run app/ui/streamlit_app.py`

On Windows, use `run_streamlit_chrome.bat` if you want the app to open in Chrome instead of the system default browser.

## Google Drive Upload
Set folder ID in `.env` or pass via CLI:

```env
GDRIVE_FOLDER_ID=your_folder_id
GDRIVE_CLIENT_SECRETS=secret/client_secret.json
GDRIVE_TOKEN=secret/token.json
```

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
- `--sample-rate <hz>`
- `--clip-resolution source|1080p|720p|480p|360p`
- `--clip-mode accurate|fast`
- `--expected-song-count <int>`
- `--merge-gap <seconds>`
- `--fingerprint-threshold <0-1>`
- `--gdrive-upload true|false`
- `--gdrive-folder-id <id>`
- `--gdrive-client-secrets <path>`
- `--gdrive-token <path>`
- `--gdrive-include-tmp true|false`

Notes:
- `--clip-resolution source` keeps original resolution.
- Presets like `720p` and `1080p` re-encode clip video and preserve aspect ratio with padding when needed.
- If `--clip-mode fast` is combined with a fixed resolution preset, clipping auto-switches to accurate mode for that clip.
- `--expected-song-count` is a merge hint that reduces over-splitting by coalescing nearest neighboring segments toward your target count.
- If finalized clip count is still higher than expected, increase `--merge-gap` (for example `--merge-gap 3.5`) to allow wider pauses to merge.
- `--outdir` points to a parent folder; each run creates a sanitized `<title>` subfolder (falls back to video ID if title is missing).
- Google Drive upload uses OAuth user login and stores a token at `secret/token.json` by default.

## Output Layout
```text
output/
  <title>/
    vods/
    audio/
    clips/
    manifests/
    previews/
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
