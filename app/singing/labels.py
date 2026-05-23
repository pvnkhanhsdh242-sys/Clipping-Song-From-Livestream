"""Reviewed manifest label loading for singing candidate training."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


TRUTHY_LABELS = {"1", "true", "t", "yes", "y", "on", "singing", "sung", "positive", "pos"}
FALSY_LABELS = {"0", "false", "f", "no", "n", "off", "not_singing", "negative", "neg", "speech", "music_only"}


@dataclass(frozen=True)
class LabeledCandidate:
    source_video: Path
    start_sec: float
    end_sec: float
    label_singing: int
    manifest_path: Path
    row_index: int
    label_quality: str | None = None
    label_notes: str | None = None
    row: dict[str, object] | None = None


def parse_label_singing(value: object) -> int | None:
    """Parse the optional manifest label into 1/0, returning None when unlabeled."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)

    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in TRUTHY_LABELS:
        return 1
    if normalized in FALSY_LABELS:
        return 0
    raise ValueError(f"Unsupported label_singing value: {value!r}")


def _resolve_source_path(raw_path: object, manifest_path: Path) -> Path:
    value = Path(str(raw_path)).expanduser()
    if value.is_absolute():
        return value
    local = (manifest_path.parent / value).resolve()
    if local.exists():
        return local
    return value.resolve()


def _load_manifest_rows(path: Path) -> list[dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return [dict(row) for row in payload if isinstance(row, dict)]
        raise ValueError(f"Manifest JSON must contain a list of rows: {path}")
    raise ValueError(f"Unsupported manifest format: {path}")


def load_labeled_candidates(manifest_paths: Iterable[Path]) -> list[LabeledCandidate]:
    """Load all rows with a valid label_singing field from reviewed manifests."""
    labeled: list[LabeledCandidate] = []
    for manifest_path in manifest_paths:
        path = manifest_path.expanduser().resolve()
        rows = _load_manifest_rows(path)
        for row_index, row in enumerate(rows, start=1):
            label = parse_label_singing(row.get("label_singing"))
            if label is None:
                continue

            source = row.get("source_video")
            start = row.get("start_sec")
            end = row.get("end_sec")
            if source in {None, ""} or start in {None, ""} or end in {None, ""}:
                raise ValueError(f"Labeled row {row_index} in {path} is missing source_video/start_sec/end_sec")

            labeled.append(
                LabeledCandidate(
                    source_video=_resolve_source_path(source, path),
                    start_sec=float(start),
                    end_sec=float(end),
                    label_singing=label,
                    manifest_path=path,
                    row_index=row_index,
                    label_quality=str(row.get("label_quality") or "").strip() or None,
                    label_notes=str(row.get("label_notes") or "").strip() or None,
                    row=row,
                )
            )
    return labeled
