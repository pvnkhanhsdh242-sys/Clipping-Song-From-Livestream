# Karaoke VOD Auto-Detect-and-Clip Setup Flow

## 1) Goal

Build an MVP pipeline that can:

1. take a **YouTube VOD URL** or a **local VOD file**,
2. **detect candidate sung-song regions**,
3. **identify the song**,
4. **refine the timestamps**,
5. **clip and save** the matching video/audio segments,
6. write a **manifest** (`.json` / `.csv`) for downstream review.

This is best treated as a **new repo** built from proven components rather than assuming there is already one turnkey repo that does everything end-to-end.

---

## 2) Recommended Stack

### Core components

- **yt-dlp** — ingest YouTube VODs and save metadata  
  Repo: https://github.com/yt-dlp/yt-dlp

- **FFmpeg** — extract audio, trim clips, re-encode as needed  
  Repo: https://github.com/FFmpeg/FFmpeg

- **inaSpeechSegmenter** — first-pass segmentation into speech / music / noise  
  Repo: https://github.com/ina-foss/inaSpeechSegmenter

- **WhisperX** — transcript + word-level timestamps for boundary refinement  
  Repo: https://github.com/m-bain/whisperx

### Song identification layer

Pick one of these for v1:

- **AudD** — external API for music recognition  
  Docs: https://audd.io/

- **ACRCloud** — external API for music recognition  
  Docs: https://www.acrcloud.com/

### Optional / fallback components

- **pyannote.audio** — stronger diarization / segmentation research stack  
  Repo: https://github.com/pyannote/pyannote-audio

- **Silero VAD** — lightweight voice activity detection  
  Repo: https://github.com/snakers4/silero-vad

- **python-lyrics-transcriber** — karaoke-style synced lyric generation / alignment ideas  
  Repo: https://github.com/karaokenerds/python-lyrics-transcriber

- **whisper-timestamped** — alternative timestamping path if we want to test against WhisperX  
  Repo: https://github.com/linto-ai/whisper-timestamped

---

## 3) Recommended Approach

### Best MVP path

Use this sequence:

**yt-dlp -> FFmpeg -> inaSpeechSegmenter -> AudD/ACRCloud -> WhisperX -> FFmpeg clipper -> manifest writer**

Why this path:

- `yt-dlp` solves YouTube VOD ingestion cleanly.
- `FFmpeg` handles all media transformations and clipping.
- `inaSpeechSegmenter` narrows the search to music-heavy sections.
- `AudD` or `ACRCloud` gives actual song identification.
- `WhisperX` helps refine the timestamps for cleaner boundaries.
- A manifest keeps the output reviewable and reproducible.

---

## 4) What We Should Build

### Deliverables for v1

The repo should deliver:

- input: **YouTube VOD URL** or **local MP4**
- output:
  - downloaded master VOD (if URL mode)
  - extracted WAV audio
  - detected clip MP4 files
  - optional detected clip WAV files
  - `manifest.json`
  - logs

### Example output structure

```text
output/
  vods/
    title_[videoid].mp4
    title_[videoid].info.json
  audio/
    title_[videoid].wav
  clips/
    song_001.mp4
    song_001.wav
    song_002.mp4
  manifests/
    title_[videoid]_manifest.json
  logs/
    run_2026-04-22.log
```

### Example manifest record

```json
{
  "video_id": "abc123",
  "song": "Example Song",
  "artist": "Example Artist",
  "start_sec": 752.2,
  "end_sec": 941.8,
  "start_tc": "00:12:32.200",
  "end_tc": "00:15:41.800",
  "confidence": 0.92,
  "clip_path": "output/clips/song_001.mp4",
  "audio_path": "output/clips/song_001.wav"
}
```

---

## 5) Proposed Repo Structure

```text
karaoke-clipper/
  README.md
  .env.example
  requirements.txt
  pyproject.toml
  app/
    main.py
    config.py
    ingest/
      youtube.py
    preprocess/
      extract_audio.py
    segment/
      music_segments.py
    identify/
      audd_client.py
      acrcloud_client.py
    align/
      whisperx_align.py
    clip/
      cutter.py
    output/
      manifest.py
      naming.py
    utils/
      ffmpeg.py
      timecode.py
      logging.py
  scripts/
    smoke_test.py
    batch_run.py
  tests/
    test_timecode.py
    test_manifest.py
  data/
    sample_segments.csv
```

---

## 6) End-to-End Flow

### Stage 1 — Ingest

#### Case A: URL input

1. accept a YouTube VOD URL,
2. run `yt-dlp`,
3. save the merged MP4,
4. save metadata sidecars.

#### Case B: Local input

1. accept a local `.mp4`,
2. skip download,
3. register the local file as the master source.

### Stage 2 — Audio extraction

1. extract audio from master VOD,
2. normalize to mono WAV,
3. store for segmentation and identification.

Suggested working audio format:

- mono
- 16 kHz or 22.05 kHz for fast first-pass work

### Stage 3 — Candidate region detection

1. run `inaSpeechSegmenter` on the WAV,
2. collect `music` and possibly `speech+music` regions,
3. merge adjacent short gaps,
4. drop tiny regions below a threshold.

Example merged candidate segment list:

```json
[
  {"start": 752.2, "end": 941.8, "label": "music"},
  {"start": 1503.5, "end": 1678.0, "label": "music"}
]
```

### Stage 4 — Song identification

For each candidate segment:

1. split into overlapping windows (ex: 10 to 15 seconds),
2. send windows to **AudD** or **ACRCloud**,
3. aggregate repeated matches,
4. keep the best-confidence song match,
5. record rough start/end.

### Stage 5 — Timestamp refinement

For the matched region:

1. extend slightly left and right,
2. run `WhisperX` on that region,
3. use transcript / word timings / lyric-style text if available,
4. tighten the clip boundaries.

Note: this step improves clip cleanliness, but we should treat it as **refinement**, not the only detection mechanism.

### Stage 6 — Clip export

Use `FFmpeg` to create:

- `clip_xxx.mp4`
- optional `clip_xxx.wav`

Two modes:

- **fast mode** — stream copy, quicker but boundaries may be less exact
- **accurate mode** — re-encode, slower but cleaner boundaries

### Stage 7 — Manifest + logs

Save:

- final clip metadata,
- source VOD reference,
- recognition confidence,
- timestamps,
- file paths,
- runtime logs.

---

## 7) Development Order

Build in this order:

### Milestone 1 — VOD ingest only

- URL -> local MP4
- local file path returned
- metadata saved

### Milestone 2 — Audio extraction only

- MP4 -> WAV
- smoke test against one known VOD

### Milestone 3 — Music segmentation only

- WAV -> candidate music regions
- export segment list as JSON

### Milestone 4 — Song recognition only

- one known region -> one recognized song
- compare AudD vs ACRCloud if needed

### Milestone 5 — Auto clipping

- recognized region -> MP4 clip + WAV clip

### Milestone 6 — Manifest writer

- all outputs consolidated into machine-readable records

### Milestone 7 — Boundary refinement

- WhisperX-based cleanup for better start/end quality

This order reduces debugging complexity and makes it easier to validate each stage.

---

## 8) Setup Instructions

### Prerequisites

Install:

- Python
- FFmpeg
- Git

### Create environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### Install packages

```bash
pip install yt-dlp ffmpeg-python inaSpeechSegmenter whisperx python-dotenv requests pandas
```

Optional:

```bash
pip install pyannote.audio
pip install whisper-timestamped
```

### Environment variables

Create `.env`:

```env
AUDD_API_TOKEN=
ACRCLOUD_HOST=
ACRCLOUD_ACCESS_KEY=
ACRCLOUD_ACCESS_SECRET=
```

For v1, use **one** recognition provider first.

---

## 9) Smoke Test Flow

Before automating everything, validate one segment end to end:

1. download one short VOD,
2. manually choose one sung section,
3. extract WAV,
4. confirm segmentation catches it,
5. confirm recognition API returns the expected song,
6. cut the clip,
7. verify saved output manually.

This helps isolate failures early.

---

## 10) Runtime Modes

### Mode A — Full automatic

Input: YouTube URL  
Output: clips + manifest

Flow:

- download VOD
- extract WAV
- detect candidate song regions
- identify songs
- refine timestamps
- clip and save

### Mode B — Local file mode

Input: local MP4  
Output: clips + manifest

Flow:

- skip download
- extract WAV
- continue pipeline

### Mode C — Known timestamps mode

Input: URL/local MP4 + manual timestamps  
Output: direct clip export

Flow:

- skip auto-detection
- clip directly

This is useful for QA and debugging.

---

## 11) Risks / Constraints

We should be realistic about accuracy.

### Will work best when:

- the song audio is clear,
- the song lasts long enough for fingerprinting,
- there is limited talking over the music,
- the backing track is not too distorted.

### Will be weaker when:

- streamer talks over the performance constantly,
- songs are extremely short snippets,
- there is heavy crowd noise,
- medleys overlap,
- the performer changes pitch/tempo substantially.

That is why the pipeline should store **confidence scores** and keep the manifest reviewable.

---

## 12) Suggested Naming Conventions

### Clip files

```text
{artist} - {song}__{start_tc}-{end_tc}__[videoid].mp4
```

Example:

```text
Taylor Swift - Love Story__00-12-32_00-15-41__[abc123].mp4
```

### Manifest files

```text
{title}__[videoid]_manifest.json
```

---

## 13) First Sprint Recommendation

### Sprint goal

Deliver a working v1 that can process **one VOD** end-to-end.

### Scope

- URL ingestion with `yt-dlp`
- local audio extraction with `FFmpeg`
- coarse music detection with `inaSpeechSegmenter`
- song recognition via **AudD** or **ACRCloud**
- auto clip creation via `FFmpeg`
- `manifest.json` output

### Defer to v2

- multiple recognition backends in parallel
- advanced lyric alignment
- UI / dashboard
- batch queueing
- cloud deployment

---

## 14) Repo / Tool Links

### Must-use

- yt-dlp: https://github.com/yt-dlp/yt-dlp
- FFmpeg: https://github.com/FFmpeg/FFmpeg
- inaSpeechSegmenter: https://github.com/ina-foss/inaSpeechSegmenter
- WhisperX: https://github.com/m-bain/whisperx

### Recognition

- AudD: https://audd.io/
- ACRCloud: https://www.acrcloud.com/

### Optional / research

- pyannote.audio: https://github.com/pyannote/pyannote-audio
- Silero VAD: https://github.com/snakers4/silero-vad
- python-lyrics-transcriber: https://github.com/karaokenerds/python-lyrics-transcriber
- whisper-timestamped: https://github.com/linto-ai/whisper-timestamped

---

## 15) Final Recommendation

Start with this exact v1 stack:

**yt-dlp + FFmpeg + inaSpeechSegmenter + AudD (or ACRCloud) + WhisperX**

That is the fastest route to a repo that can actually:

- ingest a VOD,
- auto-detect likely sung sections,
- identify the song,
- clip and save outputs,
- produce a reviewable manifest.

Once this works on one VOD reliably, extend to:

- batch processing,
- better refinement,
- human review mode,
- optional dashboard.
