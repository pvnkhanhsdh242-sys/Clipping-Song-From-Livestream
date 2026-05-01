"""Batch execution helper for multiple URLs or local files."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable


def _iter_plain_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            value = line.strip()
            if value and not value.startswith("#"):
                yield value


def _iter_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield {k: (v or "").strip() for k, v in row.items() if k}


def _build_command(
    item: str,
    outdir: Path,
    audio_clips: bool,
    use_acoustid: bool,
    ref_library: Path,
    device: str,
    clip_resolution: str,
    expected_song_count: int | None,
    min_segment: float,
    max_segment: float,
    merge_gap: float,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "app.main",
        "--outdir",
        str(outdir),
        "--audio-clips",
        "true" if audio_clips else "false",
        "--use-acoustid",
        "true" if use_acoustid else "false",
        "--ref-library",
        str(ref_library),
        "--device",
        device,
        "--clip-resolution",
        clip_resolution,
        "--min-segment",
        str(min_segment),
        "--max-segment",
        str(max_segment),
        "--merge-gap",
        str(merge_gap),
    ]

    if item.startswith("http://") or item.startswith("https://"):
        command.extend(["--url", item])
    else:
        command.extend(["--file", item])

    if expected_song_count is not None:
        command.extend(["--expected-song-count", str(expected_song_count)])

    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run karaoke-clipper for a batch of URLs/files")
    parser.add_argument("--input", required=True, help="Text or CSV file containing URLs/paths")
    parser.add_argument("--outdir", default="output/batch", help="Output directory")
    parser.add_argument("--audio-clips", action="store_true", help="Export WAV clips")
    parser.add_argument("--use-acoustid", action="store_true", help="Enable AcoustID lookup")
    parser.add_argument("--ref-library", default="data/reference_library.json", help="Local fingerprint library path")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--min-segment", type=float, default=8.0, help="Minimum segment length in seconds")
    parser.add_argument("--max-segment", type=float, default=240.0, help="Maximum segment length in seconds")
    parser.add_argument("--merge-gap", type=float, default=2.0, help="Merge candidate regions with <= this gap")
    parser.add_argument(
        "--clip-resolution",
        choices=["source", "1080p", "720p", "480p", "360p"],
        default="source",
        help="Output clip resolution preset",
    )
    parser.add_argument(
        "--expected-song-count",
        type=int,
        default=None,
        help="Hint expected number of songs and merge over-split candidates toward this count",
    )
    args = parser.parse_args()

    if args.min_segment <= 0:
        parser.error("--min-segment must be > 0")
    if args.max_segment <= args.min_segment:
        parser.error("--max-segment must be greater than --min-segment")
    if args.merge_gap < 0:
        parser.error("--merge-gap must be >= 0")

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    ref_library = Path(args.ref_library).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".csv":
        items = []
        for row in _iter_csv_rows(input_path):
            source = row.get("source") or row.get("url") or row.get("file")
            if source:
                items.append(source)
    else:
        items = list(_iter_plain_lines(input_path))

    if not items:
        print("No batch items found.")
        return 1

    failures = 0
    for index, item in enumerate(items, start=1):
        run_outdir = outdir
        cmd = _build_command(
            item=item,
            outdir=run_outdir,
            audio_clips=args.audio_clips,
            use_acoustid=args.use_acoustid,
            ref_library=ref_library,
            device=args.device,
            clip_resolution=args.clip_resolution,
            expected_song_count=args.expected_song_count,
            min_segment=args.min_segment,
            max_segment=args.max_segment,
            merge_gap=args.merge_gap,
        )

        print(f"[{index}/{len(items)}] Running: {' '.join(cmd)}")
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failures += 1
            print(f"[WARN] Item failed with exit code {completed.returncode}: {item}")

    print(f"Batch complete. total={len(items)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
