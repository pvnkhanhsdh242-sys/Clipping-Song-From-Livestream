# Smoke Test Checklist

## Single Run Validation
1. Choose source mode (URL or local MP4)
2. Run pipeline command
3. Confirm output directories created
4. Confirm working WAV exists
5. Confirm at least one clip file or empty manifest with logs (if no candidates)
6. Confirm manifest JSON and CSV exist
7. Confirm log file exists and has no unhandled traceback

## URL Mode Checks
1. Downloaded video appears in `output/vods`
2. yt-dlp metadata sidecar is present when available

## Local Mode Checks
1. Pipeline skips download cleanly
2. Source path in manifest points to local file

## Optional Lookup Checks
1. Run with `--use-acoustid false` and verify backend is `local-chromaprint` or `none`
2. Run with `--use-acoustid true` and key present to verify backend may become `acoustid`
