# Decision Log

## 2026-04-22 - API-free-first MVP
Decision: make local Chromaprint matching the default and keep external lookup optional.
Reason: avoid paid dependencies and keep project runnable offline.

## 2026-04-22 - Optional AcoustID backend
Decision: include optional AcoustID integration only behind interface and CLI flag.
Reason: preserve free/open baseline while enabling extra lookup path.

## 2026-04-22 - Graceful degradation for heavy dependencies
Decision: segmentation and refinement degrade safely if inaSpeechSegmenter or WhisperX are unavailable.
Reason: keep one-command local run practical on CPU-only environments.
