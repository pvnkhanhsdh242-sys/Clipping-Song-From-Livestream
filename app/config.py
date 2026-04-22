"""CLI and runtime configuration."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv


def str_to_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass(frozen=True)
class AppConfig:
    url: Optional[str]
    file: Optional[Path]
    outdir: Path
    audio_clips: bool
    min_segment: float
    max_segment: float
    use_acoustid: bool
    ref_library: Optional[Path]
    device: str
    sample_rate: int
    merge_gap: float
    clip_mode: str
    fingerprint_threshold: float
    acoustid_api_key: Optional[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karaoke-clipper",
        description="Detect sung-song regions and export clips from YouTube VOD or local MP4.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="YouTube VOD URL")
    source_group.add_argument("--file", help="Local MP4 file path")

    parser.add_argument("--outdir", default="output", help="Output directory root")
    parser.add_argument("--audio-clips", type=str_to_bool, default=False, help="Also export WAV clips (true/false)")
    parser.add_argument("--min-segment", type=float, default=8.0, help="Minimum segment length in seconds")
    parser.add_argument("--max-segment", type=float, default=240.0, help="Maximum segment length in seconds")
    parser.add_argument("--use-acoustid", type=str_to_bool, default=False, help="Enable optional AcoustID lookup")
    parser.add_argument("--ref-library", default="data/reference_library.json", help="Path to local fingerprint library JSON")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="WhisperX device")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Working WAV sample rate")
    parser.add_argument("--merge-gap", type=float, default=2.0, help="Merge candidate regions with <= this gap")
    parser.add_argument("--clip-mode", choices=["fast", "accurate"], default="accurate", help="FFmpeg clip mode")
    parser.add_argument(
        "--fingerprint-threshold",
        type=float,
        default=0.45,
        help="Minimum local fingerprint confidence to accept",
    )

    return parser


def load_config(argv: Optional[Sequence[str]] = None) -> AppConfig:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.min_segment <= 0:
        parser.error("--min-segment must be > 0")
    if args.max_segment <= args.min_segment:
        parser.error("--max-segment must be greater than --min-segment")

    file_path = Path(args.file).expanduser().resolve() if args.file else None
    if file_path and not file_path.exists():
        parser.error(f"Input file does not exist: {file_path}")

    return AppConfig(
        url=args.url,
        file=file_path,
        outdir=Path(args.outdir).expanduser().resolve(),
        audio_clips=args.audio_clips,
        min_segment=float(args.min_segment),
        max_segment=float(args.max_segment),
        use_acoustid=args.use_acoustid,
        ref_library=Path(args.ref_library).expanduser().resolve() if args.ref_library else None,
        device=args.device,
        sample_rate=int(args.sample_rate),
        merge_gap=float(args.merge_gap),
        clip_mode=args.clip_mode,
        fingerprint_threshold=float(args.fingerprint_threshold),
        acoustid_api_key=os.getenv("ACOUSTID_API_KEY"),
    )
