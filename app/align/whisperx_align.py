"""WhisperX timestamp refinement stage."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.identify.chromaprint_match import extract_temp_wav_segment


class WhisperXRefiner:
    """Refine segment boundaries using WhisperX when available."""

    def __init__(self, device: str, logger: logging.Logger) -> None:
        self.device = device
        self.logger = logger
        self._model: Optional[object] = None
        self._disabled = False

    def _ensure_model(self) -> None:
        if self._model is not None or self._disabled:
            return

        try:
            import whisperx  # type: ignore

            compute_type = "int8" if self.device == "cpu" else "float16"
            self._model = whisperx.load_model("tiny", self.device, compute_type=compute_type)
            self.logger.info("WhisperX model loaded for refinement")
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._disabled = True
            self.logger.warning("WhisperX not available; refinement fallback to coarse boundaries: %s", exc)

    def refine_segment(self, audio_path: Path, start_sec: float, end_sec: float, tmp_dir: Path) -> tuple[float, float]:
        """Attempt to tighten boundaries; return coarse bounds on failure."""
        if end_sec <= start_sec:
            return start_sec, end_sec

        self._ensure_model()
        if self._model is None:
            return start_sec, end_sec

        probe_start = max(0.0, start_sec - 1.0)
        probe_end = end_sec + 1.0
        probe_audio = extract_temp_wav_segment(audio_path, probe_start, probe_end, tmp_dir, self.logger)

        try:
            transcribe_fn = getattr(self._model, "transcribe", None)
            if not callable(transcribe_fn):
                return start_sec, end_sec

            result = transcribe_fn(str(probe_audio), batch_size=4)
            segments = result.get("segments", []) if isinstance(result, dict) else []
            if not segments:
                return start_sec, end_sec

            first_start = float(segments[0].get("start", 0.0))
            last_end = float(segments[-1].get("end", probe_end - probe_start))

            refined_start = max(0.0, probe_start + first_start)
            refined_end = max(refined_start + 0.1, probe_start + last_end)
            return refined_start, refined_end
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.logger.warning("WhisperX refinement failed, keeping coarse segment: %s", exc)
            return start_sec, end_sec
        finally:
            probe_audio.unlink(missing_ok=True)
