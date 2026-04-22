"""Optional AcoustID backend (free tier only)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from app.identify.chromaprint_match import extract_temp_wav_segment
from app.identify.types import MatchResult


class AcoustIDClient:
    """Optional external lookup backend for unresolved local matches."""

    def __init__(self, api_key: Optional[str], enabled: bool, logger: logging.Logger) -> None:
        self.api_key = api_key
        self.enabled = enabled and bool(api_key)
        self.logger = logger

        if enabled and not api_key:
            self.logger.warning("--use-acoustid was enabled but ACOUSTID_API_KEY is missing. Backend disabled.")

    def identify_segment(self, audio_path: Path, start_sec: float, end_sec: float, tmp_dir: Path) -> Optional[MatchResult]:
        """Resolve a segment using AcoustID lookup, if enabled."""
        if not self.enabled or not self.api_key:
            return None

        try:
            import acoustid  # type: ignore
        except ImportError:
            self.logger.warning("pyacoustid is not installed; AcoustID lookup skipped.")
            return None

        segment_audio = extract_temp_wav_segment(audio_path, start_sec, end_sec, tmp_dir, self.logger)
        best_match: Optional[MatchResult] = None

        try:
            matches = acoustid.match(self.api_key, str(segment_audio))
            for score, recording_id, title, artist in matches:
                current = MatchResult(
                    song=title or "Unknown Song",
                    artist=artist or "Unknown Artist",
                    confidence=float(score),
                    backend="acoustid",
                    track_id=recording_id,
                )
                if best_match is None or current.confidence > best_match.confidence:
                    best_match = current
        except Exception as exc:  # pragma: no cover - network dependent path
            self.logger.warning("AcoustID lookup failed: %s", exc)
        finally:
            segment_audio.unlink(missing_ok=True)

        return best_match
