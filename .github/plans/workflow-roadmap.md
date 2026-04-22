# Workflow Roadmap

## Milestones
1. M1 Ingest: URL/local source handling with retry-safe downloads
2. M2 Audio: consistent mono WAV extraction
3. M3 Segmentation: detect and merge music-like candidate regions
4. M4 Identification: local Chromaprint matching, optional AcoustID fallback
5. M5 Clipping: export MP4 and optional WAV clips
6. M6 Manifest: write JSON and CSV records with confidence and backend
7. M7 Refinement: WhisperX-based boundary tightening when available

## CI Mapping
- `ci.yml`: unit tests and static compile checks
- `smoke-test.yml`: synthetic local smoke pipeline run
- `manual-recognition.yml`: optional URL/API manual verification
