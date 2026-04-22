# Implementation Plan

## Status
Implemented MVP scaffold and core pipeline modules.

Verification snapshot:
- Unit tests: passed (5/5)
- CLI entrypoint: working (`python -m app.main --help`)
- Local smoke run: blocked on current machine due missing FFmpeg binary in PATH

## Chosen Pipeline (v1)
1. Input: YouTube VOD URL or local MP4.
2. Ingest: yt-dlp (URL mode) and FFmpeg for media handling.
3. Segmentation: inaSpeechSegmenter to find music-heavy candidate regions.
4. Identification: Chromaprint fingerprint match first, with optional AcoustID lookup.
5. Refinement: WhisperX for boundary cleanup.
6. Export: FFmpeg clip/save (MP4 and optional WAV).
7. Output: manifest + logs for review.

## Scope for This Setup Pass
- Create Python virtual environment workflow (.venv).
- Add repo bootstrap files for dependencies and env configuration.
- Create .github default structure for workflows, templates, and planning memory.

## .github Default Structure
- .github/plan.md
- .github/workflows/ci.yml
- .github/workflows/smoke-test.yml
- .github/workflows/manual-recognition.yml
- .github/ISSUE_TEMPLATE/bug_report.md
- .github/ISSUE_TEMPLATE/feature_request.md
- .github/ISSUE_TEMPLATE/milestone.md
- .github/PULL_REQUEST_TEMPLATE.md
- .github/CONTRIBUTING.md
- .github/SECURITY.md
- .github/CODEOWNERS
- .github/plans/workflow-roadmap.md
- .github/plans/decision-log.md
- .github/plans/smoke-test-checklist.md

## CI Policy
- Automatic CI: no live recognition API calls by default.
- Manual jobs: optional live provider verification via workflow_dispatch.

## Milestone Order
1. Ingest (URL/local input).
2. Audio extraction.
3. Music segmentation.
4. Fingerprint identification.
5. Clip export.
6. Manifest writer.
7. WhisperX refinement.

## Validation Targets
1. .venv creation and dependency install works on Windows.
2. FFmpeg is available in PATH.
3. CI workflows are valid and trigger correctly.
4. One short VOD smoke test produces clips + manifest + logs.

## Notes
- Keep confidence scores in manifest to support manual review.
- Keep naming deterministic for clips and manifest files.

## Implemented Files Snapshot
- app/main.py
- app/config.py
- app/ingest/youtube.py
- app/preprocess/extract_audio.py
- app/segment/music_segments.py
- app/identify/chromaprint_match.py
- app/identify/acoustid_client.py
- app/align/whisperx_align.py
- app/clip/cutter.py
- app/output/manifest.py
- scripts/build_reference_library.py
- scripts/batch_run.py
- scripts/smoke_test.py
- tests/test_timecode.py
- tests/test_manifest.py
- tests/test_segment_merge.py
- tests/test_ingest_youtube.py
