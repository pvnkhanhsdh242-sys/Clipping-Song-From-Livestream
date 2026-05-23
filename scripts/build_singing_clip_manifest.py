"""Build a singing training manifest from existing clip files."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.singing.labels import parse_label_singing


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS


def iter_clip_files(clip_dirs: Iterable[Path]) -> Iterable[Path]:
    for clip_dir in clip_dirs:
        root = clip_dir.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Clip directory does not exist: {root}")
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
                yield path


def probe_duration_sec(media_path: Path) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {media_path}: {completed.stderr.strip()}")
    duration = float(completed.stdout.strip())
    if duration <= 0:
        raise RuntimeError(f"Invalid duration for {media_path}: {duration}")
    return duration


def build_manifest_rows(clip_files: Iterable[Path], label_singing: int | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, clip_file in enumerate(clip_files, start=1):
        duration = probe_duration_sec(clip_file)
        rows.append(
            {
                "source_video": str(clip_file),
                "video_id": clip_file.parent.parent.name if clip_file.parent.parent else "",
                "song": clip_file.stem,
                "artist": "Unknown",
                "start_sec": 0.0,
                "end_sec": round(duration, 3),
                "duration_sec": round(duration, 3),
                "music_ratio": 1.0 if label_singing == 1 else 0.0,
                "fingerprint_confidence": 0.0,
                "duration_score": 0.4,
                "boundary_quality_score": 1.0,
                "merge_count": 0,
                "bridged_gap_total_sec": 0.0,
                "boundary_method": "clip_manifest",
                "label_singing": "" if label_singing is None else label_singing,
                "label_quality": "",
                "label_notes": "",
                "clip_index": index,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a training manifest from clip folders.")
    parser.add_argument(
        "--clips-dir",
        nargs="+",
        required=True,
        help="One or more folders containing clip media files.",
    )
    parser.add_argument("--output", required=True, help="Output CSV manifest path.")
    parser.add_argument(
        "--label-singing",
        default="",
        help="Optional label for all rows: 1/0/true/false/yes/no. Leave blank for manual labeling.",
    )
    args = parser.parse_args()

    label = parse_label_singing(args.label_singing)
    rows = build_manifest_rows(
        iter_clip_files(Path(value) for value in args.clips_dir),
        label,
    )
    if not rows:
        raise RuntimeError("No clip files found.")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
