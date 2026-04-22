"""Manifest writing utilities (JSON + CSV)."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from app.utils.timecode import seconds_to_timecode


@dataclass
class ManifestRecord:
    source_video: str
    video_id: str
    song: str
    artist: str
    start_sec: float
    end_sec: float
    confidence: float
    clip_path: str
    backend: str
    audio_path: Optional[str] = None

    def to_serializable(self) -> dict:
        data = asdict(self)
        data["start_tc"] = seconds_to_timecode(self.start_sec)
        data["end_tc"] = seconds_to_timecode(self.end_sec)
        return data


def write_manifests(records: Iterable[ManifestRecord], output_path_base: Path) -> Tuple[Path, Path]:
    """Write manifest records to JSON and CSV files."""
    output_path_base.parent.mkdir(parents=True, exist_ok=True)

    materialized: List[ManifestRecord] = list(records)
    rows = [record.to_serializable() for record in materialized]

    json_path = output_path_base.with_suffix(".json")
    csv_path = output_path_base.with_suffix(".csv")

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)

    fieldnames = [
        "source_video",
        "video_id",
        "song",
        "artist",
        "start_sec",
        "end_sec",
        "start_tc",
        "end_tc",
        "confidence",
        "clip_path",
        "audio_path",
        "backend",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path
