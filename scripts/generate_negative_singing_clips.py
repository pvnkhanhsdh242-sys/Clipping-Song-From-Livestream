"""Generate non-singing training clips from VOD gaps outside known singing intervals."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_singing_clip_manifest import probe_duration_sec
from app.utils.timecode import sanitize_filename_component


MEDIA_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
MANIFEST_FIELDNAMES = [
    "source_video",
    "video_id",
    "song",
    "artist",
    "start_sec",
    "end_sec",
    "duration_sec",
    "music_ratio",
    "fingerprint_confidence",
    "duration_score",
    "boundary_quality_score",
    "merge_count",
    "bridged_gap_total_sec",
    "boundary_method",
    "label_singing",
    "label_quality",
    "label_notes",
    "clip_index",
]


@dataclass(frozen=True)
class Interval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class NegativeSample:
    vod_path: Path
    start_sec: float
    end_sec: float
    output_path: Path

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def merge_intervals(
    intervals: Iterable[Interval],
    *,
    pad_sec: float,
    duration_sec: float,
) -> list[Interval]:
    padded = []
    for interval in intervals:
        start = max(0.0, interval.start - pad_sec)
        end = min(duration_sec, interval.end + pad_sec)
        if end > start:
            padded.append(Interval(start, end))

    merged: list[Interval] = []
    for interval in sorted(padded, key=lambda item: item.start):
        if not merged or interval.start > merged[-1].end:
            merged.append(interval)
            continue
        merged[-1] = Interval(merged[-1].start, max(merged[-1].end, interval.end))
    return merged


def find_gaps(
    occupied: Iterable[Interval],
    *,
    duration_sec: float,
    min_gap_sec: float,
) -> list[Interval]:
    gaps: list[Interval] = []
    cursor = 0.0
    for interval in sorted(occupied, key=lambda item: item.start):
        if interval.start - cursor >= min_gap_sec:
            gaps.append(Interval(cursor, interval.start))
        cursor = max(cursor, interval.end)
    if duration_sec - cursor >= min_gap_sec:
        gaps.append(Interval(cursor, duration_sec))
    return gaps


def resolve_local_vod_path(raw_source: str, manifest_path: Path) -> Path | None:
    if raw_source:
        candidate = Path(raw_source)
        if candidate.exists() and ".temp" not in candidate.name.lower():
            return candidate.resolve()

    run_root = manifest_path.parents[1]
    vods_dir = run_root / "vods"
    if not vods_dir.exists():
        return None

    raw_name = Path(raw_source).name if raw_source else ""
    if raw_name:
        local_candidate = vods_dir / raw_name
        if local_candidate.exists() and ".temp" not in local_candidate.name.lower():
            return local_candidate.resolve()

    vods = [
        path
        for path in sorted(vods_dir.iterdir())
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and ".temp" not in path.name.lower()
    ]
    return vods[0].resolve() if vods else None


def load_manifest_positive_intervals(manifest_path: Path) -> tuple[Path | None, list[Interval]]:
    with manifest_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    intervals: list[Interval] = []
    vod_path: Path | None = None
    for row in rows:
        start = row.get("start_sec")
        end = row.get("end_sec")
        if start in {None, ""} or end in {None, ""}:
            continue
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        intervals.append(Interval(start_f, end_f))
        if vod_path is None:
            vod_path = resolve_local_vod_path(str(row.get("source_video") or ""), manifest_path)
    return vod_path, intervals


def discover_positive_sources(output_root: Path) -> list[tuple[Path, Path, list[Interval]]]:
    sources: list[tuple[Path, Path, list[Interval]]] = []
    for manifest_path in sorted(output_root.rglob("*_manifest.csv")):
        if "training_smoke" in manifest_path.parts:
            continue
        vod_path, intervals = load_manifest_positive_intervals(manifest_path)
        if vod_path is None or not intervals:
            continue
        sources.append((manifest_path, vod_path, intervals))
    return sources


def plan_negative_samples(
    sources: Iterable[tuple[Path, Path, list[Interval]]],
    *,
    output_dir: Path,
    positive_pad_sec: float,
    min_negative_sec: float,
    max_sample_duration_sec: float | None,
    max_negatives: int | None,
    seed: int,
) -> list[NegativeSample]:
    rng = random.Random(seed)
    samples: list[NegativeSample] = []

    for manifest_path, vod_path, positives in sources:
        vod_duration = probe_duration_sec(vod_path)
        occupied = merge_intervals(positives, pad_sec=positive_pad_sec, duration_sec=vod_duration)
        gaps = find_gaps(occupied, duration_sec=vod_duration, min_gap_sec=min_negative_sec)
        available = [[gap.start, gap.end] for gap in gaps]
        durations = []
        for interval in positives:
            duration = max(min_negative_sec, interval.duration)
            if max_sample_duration_sec is not None:
                duration = min(duration, max_sample_duration_sec)
            durations.append(duration)
        rng.shuffle(durations)

        for target_duration in durations:
            if max_negatives is not None and len(samples) >= max_negatives:
                return samples

            fitting = [idx for idx, gap in enumerate(available) if gap[1] - gap[0] >= min(target_duration, gap[1] - gap[0])]
            if not fitting:
                continue

            # Prefer a gap that fits the full matched duration; otherwise use the longest available gap.
            full_fits = [idx for idx in fitting if available[idx][1] - available[idx][0] >= target_duration]
            gap_idx = rng.choice(full_fits) if full_fits else max(fitting, key=lambda idx: available[idx][1] - available[idx][0])
            gap_start, gap_end = available[gap_idx]
            duration = min(target_duration, gap_end - gap_start)
            if duration < min_negative_sec:
                continue

            start_max = gap_end - duration
            start = rng.uniform(gap_start, start_max) if start_max > gap_start else gap_start
            end = start + duration
            available[gap_idx] = [end, gap_end]

            run_label = manifest_path.parents[1].name
            output_name = f"{sanitize_filename_component(run_label)}_negative_{len(samples) + 1:03d}.mp4"
            samples.append(
                NegativeSample(
                    vod_path=vod_path,
                    start_sec=round(start, 3),
                    end_sec=round(end, 3),
                    output_path=output_dir / output_name,
                )
            )

    return samples


def export_negative_clip(sample: NegativeSample) -> None:
    sample.output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{sample.start_sec:.3f}",
        "-to",
        f"{sample.end_sec:.3f}",
        "-i",
        str(sample.vod_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        str(sample.output_path),
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
        raise RuntimeError(f"ffmpeg failed for {sample.output_path}: {completed.stderr.strip()}")


def write_negative_manifest(samples: Iterable[NegativeSample], output_path: Path) -> int:
    rows = []
    for index, sample in enumerate(samples, start=1):
        rows.append(
            {
                "source_video": str(sample.output_path),
                "video_id": sample.vod_path.stem,
                "song": sample.output_path.stem,
                "artist": "Unknown",
                "start_sec": 0.0,
                "end_sec": round(sample.duration_sec, 3),
                "duration_sec": round(sample.duration_sec, 3),
                "music_ratio": 0.0,
                "fingerprint_confidence": 0.0,
                "duration_score": 0.4,
                "boundary_quality_score": 1.0,
                "merge_count": 0,
                "bridged_gap_total_sec": 0.0,
                "boundary_method": "vod_gap_negative",
                "label_singing": 0,
                "label_quality": "auto_gap",
                "label_notes": f"source={sample.vod_path}; source_start={sample.start_sec}; source_end={sample.end_sec}",
                "clip_index": index,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Generate negative singing training clips from VOD gaps.")
    parser.add_argument("--output-root", default="output", help="Root folder containing run outputs.")
    parser.add_argument(
        "--negative-dir",
        default="data/training_clips/not_singing/auto",
        help="Directory for generated negative clips.",
    )
    parser.add_argument(
        "--manifest-output",
        default="output/singing_clip_train_negative.csv",
        help="Output negative manifest CSV.",
    )
    parser.add_argument("--positive-pad-sec", type=float, default=15.0, help="Safety padding around singing intervals.")
    parser.add_argument("--min-negative-sec", type=float, default=30.0, help="Minimum negative clip duration.")
    parser.add_argument(
        "--max-sample-duration-sec",
        type=float,
        default=None,
        help="Optional cap for generated negative clip duration; useful for smoke tests.",
    )
    parser.add_argument("--max-negatives", type=int, default=None, help="Maximum number of negative clips to generate.")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic sampling seed.")
    parser.add_argument("--dry-run", action="store_true", help="Plan samples without cutting clips or writing manifest.")
    args = parser.parse_args()

    if args.positive_pad_sec < 0:
        parser.error("--positive-pad-sec must be >= 0")
    if args.min_negative_sec <= 0:
        parser.error("--min-negative-sec must be > 0")
    if args.max_sample_duration_sec is not None and args.max_sample_duration_sec < args.min_negative_sec:
        parser.error("--max-sample-duration-sec must be >= --min-negative-sec")
    if args.max_negatives is not None and args.max_negatives <= 0:
        parser.error("--max-negatives must be > 0")

    sources = discover_positive_sources(Path(args.output_root).expanduser().resolve())
    if not sources:
        raise RuntimeError("No positive manifests with resolvable VODs found.")

    samples = plan_negative_samples(
        sources,
        output_dir=Path(args.negative_dir).expanduser().resolve(),
        positive_pad_sec=args.positive_pad_sec,
        min_negative_sec=args.min_negative_sec,
        max_sample_duration_sec=args.max_sample_duration_sec,
        max_negatives=args.max_negatives,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError("No negative samples could be planned from VOD gaps.")

    print(f"Planned {len(samples)} negative clips from {len(sources)} positive manifest(s).")
    if args.dry_run:
        for sample in samples[:10]:
            print(f"DRY {sample.vod_path.name}: {sample.start_sec:.3f} -> {sample.end_sec:.3f}")
        return 0

    for sample in samples:
        export_negative_clip(sample)
    count = write_negative_manifest(samples, Path(args.manifest_output).expanduser().resolve())
    print(f"Wrote {count} negative clips to {Path(args.negative_dir).expanduser().resolve()}")
    print(f"Wrote manifest to {Path(args.manifest_output).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
