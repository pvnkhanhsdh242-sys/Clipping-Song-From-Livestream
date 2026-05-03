"""CLI and runtime configuration."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv


CLIP_RESOLUTION_CHOICES = ["source", "1080p", "720p", "480p", "360p"]


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
    expected_song_count: Optional[int]
    clip_mode: str
    clip_resolution: str
    fingerprint_threshold: float
    acoustid_api_key: Optional[str]
    gdrive_upload: bool
    gdrive_folder_id: Optional[str]
    gdrive_client_secrets: Optional[Path]
    gdrive_token_path: Path
    gdrive_include_tmp: bool
    exclude_start_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karaoke-clipper",
        description="Detect sung-song regions and export clips from YouTube VOD or local MP4.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="YouTube VOD URL")
    source_group.add_argument("--file", help="Local MP4 file path")

    parser.add_argument(
        "--outdir",
        default="output",
        help="Parent output directory (run subfolder named after source title)",
    )
    parser.add_argument("--audio-clips", type=str_to_bool, default=False, help="Also export WAV clips (true/false)")
    parser.add_argument("--min-segment", type=float, default=8.0, help="Minimum segment length in seconds")
    parser.add_argument("--max-segment", type=float, default=240.0, help="Maximum segment length in seconds")
    parser.add_argument("--use-acoustid", type=str_to_bool, default=False, help="Enable optional AcoustID lookup")
    parser.add_argument("--ref-library", default="data/reference_library.json", help="Path to local fingerprint library JSON")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="WhisperX device")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Working WAV sample rate")
    parser.add_argument("--merge-gap", type=float, default=2.0, help="Merge candidate regions with <= this gap")
    parser.add_argument(
        "--expected-song-count",
        type=int,
        default=None,
        help="Hint expected number of songs and merge over-split candidates toward this count",
    )
    parser.add_argument("--clip-mode", choices=["fast", "accurate"], default="accurate", help="FFmpeg clip mode")
    parser.add_argument(
        "--clip-resolution",
        choices=CLIP_RESOLUTION_CHOICES,
        default="source",
        help="Output clip resolution preset",
    )
    parser.add_argument(
        "--fingerprint-threshold",
        type=float,
        default=0.45,
        help="Minimum local fingerprint confidence to accept",
    )
    parser.add_argument(
        "--gdrive-upload",
        type=str_to_bool,
        default=False,
        help="Upload output folder to Google Drive (true/false)",
    )
    parser.add_argument(
        "--gdrive-folder-id",
        default=None,
        help="Google Drive folder ID to upload into",
    )
    parser.add_argument(
        "--gdrive-client-secrets",
        default=None,
        help="Path to Google OAuth client secrets JSON",
    )
    parser.add_argument(
        "--gdrive-token",
        default="secret/token.json",
        help="Path to cache Google OAuth token",
    )
    parser.add_argument(
        "--gdrive-include-tmp",
        type=str_to_bool,
        default=False,
        help="Include tmp folder in Google Drive upload (true/false)",
    )

    parser.add_argument(
        "--exclude-start-seconds",
        type=float,
        default=0.0,
        help="Seconds at start of audio to ignore for segmentation",
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
    if args.expected_song_count is not None and args.expected_song_count <= 0:
        parser.error("--expected-song-count must be > 0")

    file_path = Path(args.file).expanduser().resolve() if args.file else None
    if file_path and not file_path.exists():
        parser.error(f"Input file does not exist: {file_path}")

    gdrive_folder_id = args.gdrive_folder_id or os.getenv("GDRIVE_FOLDER_ID")
    gdrive_client_secrets = (
        args.gdrive_client_secrets
        or os.getenv("GDRIVE_CLIENT_SECRETS")
        or "secret"
    )
    gdrive_token_value = os.getenv("GDRIVE_TOKEN") or args.gdrive_token

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
        expected_song_count=args.expected_song_count,
        clip_mode=args.clip_mode,
        clip_resolution=args.clip_resolution,
        fingerprint_threshold=float(args.fingerprint_threshold),
        acoustid_api_key=os.getenv("ACOUSTID_API_KEY"),
        gdrive_upload=bool(args.gdrive_upload),
        gdrive_folder_id=str(gdrive_folder_id).strip() if gdrive_folder_id else None,
        gdrive_client_secrets=(
            Path(gdrive_client_secrets).expanduser().resolve()
            if gdrive_client_secrets
            else None
        ),
        gdrive_token_path=Path(gdrive_token_value).expanduser().resolve(),
        gdrive_include_tmp=bool(args.gdrive_include_tmp),
        exclude_start_seconds=float(args.exclude_start_seconds),
    )
