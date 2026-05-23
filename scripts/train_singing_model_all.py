"""Cross-platform orchestration for singing model training."""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.singing.training import train_singing_candidate_model
from scripts.build_singing_clip_manifest import MEDIA_EXTENSIONS, build_manifest_rows, iter_clip_files
from scripts.generate_negative_singing_clips import (
    discover_positive_sources,
    export_negative_clip,
    plan_negative_samples,
    write_negative_manifest,
)


def parse_bool(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def discover_positive_clip_dirs(output_root: Path) -> list[Path]:
    root = output_root.expanduser().resolve()
    if not root.exists():
        return []
    dirs: list[Path] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        clips_dir = run_dir / "clips"
        if not clips_dir.exists() or clips_dir.name.lower() == "clips_old":
            continue
        has_media = any(
            path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
            for path in clips_dir.iterdir()
        )
        if has_media:
            dirs.append(clips_dir.resolve())
    return dirs


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def count_media_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(
        1
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    )


def write_clip_manifest(clip_dirs: Iterable[Path], output_path: Path, label_singing: int) -> int:
    rows = build_manifest_rows(iter_clip_files(clip_dirs), label_singing)
    if not rows:
        raise RuntimeError("No clip files found for manifest generation.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _safe_remove_auto_negative_dir(auto_negative_dir: Path, negative_dir: Path) -> None:
    target = auto_negative_dir.expanduser().resolve()
    root = negative_dir.expanduser().resolve()
    if not target.exists():
        return
    if target == root or root not in target.parents:
        raise RuntimeError(f"Refusing to remove unexpected path: {target}")
    shutil.rmtree(target)


def generate_negatives(
    *,
    output_root: Path,
    negative_dir: Path,
    auto_negative_dir: Path,
    negative_manifest: Path,
    needed_count: int,
    positive_pad_sec: float,
    min_negative_sec: float,
    max_negative_duration_sec: float | None,
    random_state: int,
    dry_run: bool,
) -> int:
    sources = discover_positive_sources(output_root.expanduser().resolve())
    if not sources:
        raise RuntimeError("No positive manifests with resolvable VODs found for automatic negative generation.")

    samples = plan_negative_samples(
        sources,
        output_dir=auto_negative_dir.expanduser().resolve(),
        positive_pad_sec=positive_pad_sec,
        min_negative_sec=min_negative_sec,
        max_sample_duration_sec=max_negative_duration_sec,
        max_negatives=needed_count,
        seed=random_state,
    )
    if not samples:
        raise RuntimeError("No automatic negative clips could be planned from VOD gaps.")

    if dry_run:
        print(f"[dry-run] Would generate {len(samples)} automatic negative clips.")
        return len(samples)

    _safe_remove_auto_negative_dir(auto_negative_dir, negative_dir)
    for sample in samples:
        export_negative_clip(sample)
    return write_negative_manifest(samples, negative_manifest)


def find_first_vod(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    for path in sorted(output_root.rglob("*.mp4")):
        if "vods" in {part.lower() for part in path.parts}:
            return path.resolve()
    return None


def run_score_evaluation(
    *,
    eval_file: Path | None,
    output_root: Path,
    eval_outdir: Path,
    model_dir: Path,
    threshold: float,
) -> int:
    target = eval_file.expanduser().resolve() if eval_file else find_first_vod(output_root)
    if target is None:
        print("No VOD MP4 found under output/*/vods; skipping score-mode evaluation.")
        return 0
    command = [
        sys.executable,
        "-m",
        "app.main",
        "--file",
        str(target),
        "--outdir",
        str(eval_outdir),
        "--singing-model-mode",
        "score",
        "--singing-model-path",
        str(model_dir),
        "--singing-score-threshold",
        str(threshold),
    ]
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build manifests, generate negatives, and train a singing scorer.")
    parser.add_argument("--output-root", default="output", help="Root folder containing run outputs.")
    parser.add_argument("--positive-manifest", default="output/singing_clip_train_positive.csv")
    parser.add_argument("--negative-manifest", default="output/singing_clip_train_negative.csv")
    parser.add_argument("--negative-dir", default="data/training_clips/not_singing")
    parser.add_argument("--auto-negative-dir", default="data/training_clips/not_singing/auto")
    parser.add_argument("--model-dir", default="data/models/singing_candidate")
    parser.add_argument("--backend", choices=["sklearn", "pytorch"], default="sklearn")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--window-sec", type=float, default=12.0)
    parser.add_argument("--windows-per-clip", type=int, default=4)
    parser.add_argument("--validation-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=13)
    parser.add_argument("--positive-pad-sec", type=float, default=15.0)
    parser.add_argument("--min-negative-sec", type=float, default=30.0)
    parser.add_argument("--max-negative-duration-sec", type=float, default=None)
    parser.add_argument("--max-negatives", type=int, default=None)
    parser.add_argument("--run-eval", type=parse_bool, default=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--eval-outdir", default="output/eval")
    parser.add_argument("--singing-score-threshold", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", help="Plan work without writing manifests or training.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.window_sec <= 0:
        parser.error("--window-sec must be > 0")
    if args.windows_per_clip <= 0:
        parser.error("--windows-per-clip must be > 0")
    if args.max_negatives is not None and args.max_negatives <= 0:
        parser.error("--max-negatives must be > 0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_root = Path(args.output_root)
    positive_manifest = Path(args.positive_manifest)
    negative_manifest = Path(args.negative_manifest)
    negative_dir = Path(args.negative_dir)
    auto_negative_dir = Path(args.auto_negative_dir)
    model_dir = Path(args.model_dir)

    print("[1/5] Discovering positive clips...")
    positive_dirs = discover_positive_clip_dirs(output_root)
    if not positive_dirs:
        raise RuntimeError("No non-empty output/*/clips folders found.")
    print(f"Found {len(positive_dirs)} positive clip folder(s).")

    if args.dry_run:
        print(f"[dry-run] Would write positive manifest: {positive_manifest}")
        positive_count = sum(count_media_files(path) for path in positive_dirs)
    else:
        print("[2/5] Building positive manifest...")
        positive_count = write_clip_manifest(positive_dirs, positive_manifest, 1)
    print(f"Positive clips: {positive_count}")

    existing_negative_count = count_media_files(negative_dir)
    needed_negative_count = max(0, int(positive_count) - int(existing_negative_count))
    if args.max_negatives is not None and needed_negative_count > 0:
        needed_negative_count = min(needed_negative_count, int(args.max_negatives))

    print("[3/5] Preparing negative clips...")
    if needed_negative_count > 0:
        print(f"Need {needed_negative_count} more negative clip(s); generating from VOD gaps.")
        generated = generate_negatives(
            output_root=output_root,
            negative_dir=negative_dir,
            auto_negative_dir=auto_negative_dir,
            negative_manifest=negative_manifest,
            needed_count=needed_negative_count,
            positive_pad_sec=float(args.positive_pad_sec),
            min_negative_sec=float(args.min_negative_sec),
            max_negative_duration_sec=args.max_negative_duration_sec,
            random_state=int(args.random_state),
            dry_run=bool(args.dry_run),
        )
        print(f"Automatic negative clips: {generated}")
    else:
        print(f"Existing negative clips: {existing_negative_count}")

    total_negative_count = count_media_files(negative_dir)
    if args.dry_run:
        print(f"[dry-run] Would write negative manifest: {negative_manifest}")
        print(f"[dry-run] Would train {args.backend} model into: {model_dir}")
        return 0

    if total_negative_count <= 0:
        raise RuntimeError("No negative clips available. Add negatives or allow automatic VOD-gap generation.")

    negative_count = write_clip_manifest([negative_dir], negative_manifest, 0)
    print(f"Negative clips: {negative_count}")

    print("[4/5] Training model...")
    result = train_singing_candidate_model(
        [positive_manifest, negative_manifest],
        model_dir,
        backend=args.backend,
        validation_size=float(args.validation_size),
        random_state=int(args.random_state),
        max_iter=int(args.epochs),
        device=args.device,
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        window_sec=float(args.window_sec),
        windows_per_clip=int(args.windows_per_clip),
        logger=logging.getLogger("karaoke_clipper.singing_training"),
    )
    print(f"Wrote model: {result.model_path}")
    print(f"Wrote metadata: {result.metadata_path}")
    print(
        "Labels: "
        f"total={result.labeled_count} positive={result.positive_count} negative={result.negative_count}"
    )
    print(f"Metrics: {result.metrics}")

    if args.run_eval:
        print("[5/5] Running score-mode evaluation...")
        code = run_score_evaluation(
            eval_file=Path(args.eval_file) if args.eval_file else None,
            output_root=output_root,
            eval_outdir=Path(args.eval_outdir),
            model_dir=model_dir,
            threshold=float(args.singing_score_threshold),
        )
        if code != 0:
            return code
    else:
        print("[5/5] Score-mode evaluation skipped.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
