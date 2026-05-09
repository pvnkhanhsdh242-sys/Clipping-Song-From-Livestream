"""WhisperX timestamp refinement stage."""

from __future__ import annotations

import logging
import wave
from array import array
from pathlib import Path
from typing import Optional

from app.identify.chromaprint_match import extract_temp_wav_segment


def _build_audio_payload(probe_audio: Path, logger: logging.Logger) -> dict[str, object] | None:
    """Load mono waveform tensor dict to bypass torchcodec file decoding."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch should exist when whisperx works
        logger.warning("Torch unavailable for WhisperX in-memory audio payload: %s", exc)
        return None

    try:
        with wave.open(str(probe_audio), "rb") as wav_handle:
            sample_rate = wav_handle.getframerate()
            channels = wav_handle.getnchannels()
            sample_width = wav_handle.getsampwidth()
            frame_count = wav_handle.getnframes()
            payload = wav_handle.readframes(frame_count)

        if sample_width != 2:
            logger.warning("WhisperX payload expects 16-bit PCM WAV; got %s-byte samples", sample_width)
            return None

        samples = array("h")
        samples.frombytes(payload)

        if channels > 1:
            mono = array("h")
            for idx in range(0, len(samples), channels):
                frame = samples[idx : idx + channels]
                mono.append(int(sum(frame) / len(frame)))
            samples = mono

        waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
        waveform = waveform.unsqueeze(0)
        return {"waveform": waveform, "sample_rate": int(sample_rate)}
    except Exception as exc:  # pragma: no cover - depends on input file
        logger.warning("Failed to build WhisperX in-memory audio payload: %s", exc)
        return None


def _safe_refine_music_boundary(
    original_start: float,
    original_end: float,
    whisper_start: float,
    whisper_end: float,
    max_start_shrink_sec: float,
    max_end_shrink_sec: float,
    post_roll_sec: float,
    audio_duration_sec: float | None,
) -> tuple[float, float]:
    if whisper_start > original_start + max_start_shrink_sec:
        final_start = original_start
    else:
        final_start = min(original_start, whisper_start)

    if whisper_end < original_end - max_end_shrink_sec:
        final_end = original_end
    else:
        final_end = max(original_end, whisper_end)

    final_end += post_roll_sec
    if audio_duration_sec is not None:
        final_end = min(final_end, audio_duration_sec)

    if final_end <= final_start:
        final_end = final_start + 0.1

    return final_start, final_end


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

    def refine_segment(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
        tmp_dir: Path,
        mode: str = "safe",
        max_start_shrink_sec: float = 0.5,
        max_end_shrink_sec: float = 0.5,
        post_roll_sec: float = 0.0,
        audio_duration_sec: float | None = None,
    ) -> tuple[float, float]:
        """Attempt to refine boundaries; return coarse bounds on failure."""
        if end_sec <= start_sec:
            return start_sec, end_sec

        if mode == "off":
            return start_sec, end_sec

        self._ensure_model()
        if self._model is None:
            return start_sec, end_sec

        probe_start = max(0.0, start_sec - 1.0)
        probe_end = end_sec + 1.0
        probe_audio = extract_temp_wav_segment(audio_path, probe_start, probe_end, tmp_dir, self.logger)
        probe_payload = _build_audio_payload(probe_audio, self.logger)
        if probe_payload is None:
            return start_sec, end_sec

        try:
            transcribe_fn = getattr(self._model, "transcribe", None)
            if not callable(transcribe_fn):
                return start_sec, end_sec

            try:
                result = transcribe_fn(
                    probe_payload,
                    batch_size=4,
                    vad_filter=False,
                )
            except TypeError as exc:
                if "unexpected keyword argument 'vad_filter'" not in str(exc):
                    raise
                # Avoid implicit VAD/decoder paths on older builds that don't support vad_filter.
                self.logger.warning(
                    "WhisperX build does not support vad_filter; skipping boundary refinement to avoid decoder/VAD issues."
                )
                return start_sec, end_sec
            segments = result.get("segments", []) if isinstance(result, dict) else []
            if not segments:
                return start_sec, end_sec

            first_start = float(segments[0].get("start", 0.0))
            last_end = float(segments[-1].get("end", probe_end - probe_start))

            whisper_start = max(0.0, probe_start + first_start)
            whisper_end = max(whisper_start + 0.1, probe_start + last_end)

            if mode == "metadata":
                return start_sec, end_sec

            refined_start, refined_end = _safe_refine_music_boundary(
                original_start=start_sec,
                original_end=end_sec,
                whisper_start=whisper_start,
                whisper_end=whisper_end,
                max_start_shrink_sec=max_start_shrink_sec,
                max_end_shrink_sec=max_end_shrink_sec,
                post_roll_sec=post_roll_sec,
                audio_duration_sec=audio_duration_sec,
            )
            return refined_start, refined_end
        except Exception as exc:  # pragma: no cover - optional dependency path
            self.logger.warning("WhisperX refinement failed, keeping coarse segment: %s", exc)
            return start_sec, end_sec
        finally:
            probe_audio.unlink(missing_ok=True)
