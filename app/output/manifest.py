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
    raw_start_sec: float
    raw_end_sec: float
    start_sec: float
    end_sec: float
    duration_sec: float
    pre_roll_sec: float
    post_roll_sec: float
    boundary_method: str
    refinement_method: str
    music_ratio: float
    fingerprint_confidence: float
    duration_score: float
    boundary_quality_score: float
    final_score: float
    merge_count: int
    bridged_gap_total_sec: float
    needs_review: bool
    review_reason: Optional[str]
    confidence: float
    clip_path: str
    backend: str
    audio_path: Optional[str] = None
    singing_score: Optional[float] = None
    singing_model: str = "none"
    singing_decision: str = "not_scored"
    label_singing: Optional[str] = None
    label_quality: Optional[str] = None
    label_notes: Optional[str] = None

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
        "raw_start_sec",
        "raw_end_sec",
        "start_sec",
        "end_sec",
        "duration_sec",
        "start_tc",
        "end_tc",
        "pre_roll_sec",
        "post_roll_sec",
        "boundary_method",
        "refinement_method",
        "music_ratio",
        "fingerprint_confidence",
        "duration_score",
        "boundary_quality_score",
        "final_score",
        "merge_count",
        "bridged_gap_total_sec",
        "needs_review",
        "review_reason",
        "confidence",
        "clip_path",
        "audio_path",
        "backend",
        "singing_score",
        "singing_model",
        "singing_decision",
        "label_singing",
        "label_quality",
        "label_notes",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path
