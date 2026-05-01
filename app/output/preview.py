"""Preview record helpers for UI and dry runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from app.utils.timecode import seconds_to_timecode
from app.utils.ffmpeg import has_video_stream, run_command


@dataclass(frozen=True)
class PreviewRecord:
    index: int
    start_sec: float
    end_sec: float
    song: str
    artist: str
    confidence: float
    backend: str

    def to_row(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "start_tc": seconds_to_timecode(self.start_sec),
            "end_tc": seconds_to_timecode(self.end_sec),
            "song": self.song,
            "artist": self.artist,
            "confidence": round(self.confidence, 4),
            "backend": self.backend,
        }


def _snapshot_path(output_dir: Path, record: PreviewRecord) -> Path:
    return output_dir / f"segment_{record.index:03d}_{int(record.start_sec):06d}.jpg"


def generate_snapshots(
    video_path: Path,
    records: Iterable[PreviewRecord],
    output_dir: Path,
    logger,
    limit: int = 0,
) -> List[Path]:
    """Generate timestamp screenshots for preview records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshots: List[Path] = []

    if not has_video_stream(video_path, logger):
        logger.warning("Preview snapshots skipped; no video stream in %s", video_path)
        return snapshots

    for idx, record in enumerate(records):
        if limit and idx >= limit:
            break

        target = _snapshot_path(output_dir, record)
        if not target.exists():
            command = [
                "ffmpeg",
                "-y",
                "-ss",
                f"{record.start_sec:.3f}",
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(target),
            ]
            result = run_command(command, logger=logger, check=False)
            if result.returncode != 0 or not target.exists():
                logger.warning("Snapshot failed for segment %s", record.index)
                continue
        if target.exists():
            snapshots.append(target)

    return snapshots
