"""Local Chromaprint-based matching backend."""

from __future__ import annotations

import json
import logging
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from app.identify.types import MatchResult
from app.utils.ffmpeg import run_command


def extract_temp_wav_segment(
    source_audio: Path,
    start_sec: float,
    end_sec: float,
    tmp_dir: Path,
    logger: logging.Logger,
) -> Path:
    """Extract a temporary WAV segment used for fingerprinting."""
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")

    tmp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir, delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        "11025",
        "-c:a",
        "pcm_s16le",
        str(temp_path),
    ]
    run_command(command, logger=logger)
    return temp_path


def fingerprint_audio_file(audio_path: Path) -> tuple[float, str]:
    """Compute a Chromaprint fingerprint using pyacoustid/fpcalc."""
    try:
        import acoustid  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyacoustid is required for fingerprinting") from exc

    duration, fingerprint = acoustid.fingerprint_file(str(audio_path))
    return float(duration), str(fingerprint)


def _normalize_score(raw_score: Any) -> float:
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0

    if score < 0:
        return 0.0
    if score <= 1.0:
        return score
    return 1.0 / (1.0 + score)


def compare_fingerprints(query_fp: str, candidate_fp: str) -> float:
    """Compare two fingerprints and return confidence in [0, 1]."""
    try:
        import acoustid  # type: ignore

        compare_fn = getattr(acoustid, "compare_fingerprints", None)
        if callable(compare_fn):
            raw = compare_fn(query_fp, candidate_fp)
            if isinstance(raw, tuple) and raw:
                raw = raw[0]
            return _normalize_score(raw)
    except Exception:
        pass

    return SequenceMatcher(None, query_fp, candidate_fp).ratio()


class ChromaprintMatcher:
    """Match candidate segments against a local fingerprint library."""

    def __init__(self, library_path: Optional[Path], threshold: float, logger: logging.Logger) -> None:
        self.library_path = library_path
        self.threshold = threshold
        self.logger = logger
        self.records = self._load_library()

    def _load_library(self) -> list[dict[str, Any]]:
        if not self.library_path:
            self.logger.info("No local fingerprint library configured")
            return []

        if not self.library_path.exists():
            self.logger.warning("Reference library not found: %s", self.library_path)
            return []

        with self.library_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        tracks = payload.get("tracks", payload if isinstance(payload, list) else [])
        records: list[dict[str, Any]] = []

        for idx, item in enumerate(tracks):
            fingerprint = item.get("fingerprint")
            if not fingerprint:
                continue

            records.append(
                {
                    "track_id": item.get("track_id", f"local-{idx}"),
                    "title": item.get("title", "Unknown Song"),
                    "artist": item.get("artist", "Unknown Artist"),
                    "fingerprint": str(fingerprint),
                    "duration": float(item.get("duration", 0.0)),
                }
            )

        self.logger.info("Loaded %s fingerprint records from %s", len(records), self.library_path)
        return records

    def match_segment(self, audio_path: Path, start_sec: float, end_sec: float, tmp_dir: Path) -> Optional[MatchResult]:
        """Match one candidate segment against the local library."""
        if not self.records:
            return None

        segment_audio = extract_temp_wav_segment(audio_path, start_sec, end_sec, tmp_dir, self.logger)
        try:
            _, query_fingerprint = fingerprint_audio_file(segment_audio)
        except Exception as exc:
            self.logger.warning("Fingerprint generation failed: %s", exc)
            try:
                segment_audio.unlink(missing_ok=True)
            except Exception:
                pass
            return None

        best_score = 0.0
        best_record: Optional[dict[str, Any]] = None

        for record in self.records:
            score = compare_fingerprints(query_fingerprint, record["fingerprint"])
            if score > best_score:
                best_score = score
                best_record = record

        segment_audio.unlink(missing_ok=True)

        if not best_record or best_score < self.threshold:
            return None

        return MatchResult(
            song=str(best_record["title"]),
            artist=str(best_record["artist"]),
            confidence=round(best_score, 4),
            backend="local-chromaprint",
            track_id=str(best_record.get("track_id")) if best_record.get("track_id") else None,
        )
