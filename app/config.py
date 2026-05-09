"""CLI and runtime configuration."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv


CLIP_RESOLUTION_CHOICES = ["source", "1080p", "720p", "480p", "360p"]
WHISPERX_BOUNDARY_MODES = ["off", "metadata", "safe"]
PROFILE_CHOICES = ["karaoke", "concert", "mixed_stream", "strict", "custom"]
RUNTIME_DEVICE_ENV = "KARAOKE_FORCE_DEVICE"

PROFILES: dict[str, dict[str, float | str]] = {
    "karaoke": {
        "min_segment": 30.0,
        "max_segment": 420.0,
        "merge_gap": 1.5,
        "bridge_noise_gap": 2.0,
        "bridge_speech_gap": 1.0,
        "pre_roll": 0.5,
        "post_roll": 2.0,
        "whisperx_boundary_mode": "safe",
    },
    "concert": {
        "min_segment": 45.0,
        "max_segment": 600.0,
        "merge_gap": 2.5,
        "bridge_noise_gap": 3.0,
        "bridge_speech_gap": 1.5,
        "pre_roll": 1.0,
        "post_roll": 3.0,
        "whisperx_boundary_mode": "safe",
    },
    "mixed_stream": {
        "min_segment": 20.0,
        "max_segment": 420.0,
        "merge_gap": 1.0,
        "bridge_noise_gap": 1.5,
        "bridge_speech_gap": 0.8,
        "pre_roll": 0.5,
        "post_roll": 2.0,
        "whisperx_boundary_mode": "metadata",
    },
    "strict": {
        "min_segment": 45.0,
        "max_segment": 360.0,
        "merge_gap": 0.8,
        "bridge_noise_gap": 0.8,
        "bridge_speech_gap": 0.3,
        "pre_roll": 0.3,
        "post_roll": 1.0,
        "whisperx_boundary_mode": "off",
    },
}


def resolve_runtime_device(preferred_device: str) -> str:
    forced_device = os.getenv(RUNTIME_DEVICE_ENV, "").strip().lower()
    if forced_device in {"cpu", "cuda"}:
        return forced_device
    return preferred_device


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
    merge_max_segment: float
    segment_tolerance: float
    pre_roll_sec: float
    post_roll_sec: float
    bridge_noise_gap_sec: float
    bridge_speech_gap_sec: float
    expected_song_count: Optional[int]
    clip_mode: str
    clip_resolution: str
    fingerprint_threshold: float
    acoustid_api_key: Optional[str]
    whisperx_boundary_mode: str
    whisperx_max_start_shrink_sec: float
    whisperx_max_end_shrink_sec: float
    allow_hard_split: bool
    energy_frame_ms: int
    energy_min_active_ms: int
    energy_min_silence_ms: int
    profile: str
    review_score_threshold: float
    gdrive_upload: bool
    gdrive_folder_id: Optional[str]
    gdrive_client_secrets: Optional[Path]
    gdrive_token_path: Path
    gdrive_include_tmp: bool
    gdrive_upload_mode: str
    exclude_start_seconds: float
    exclude_end_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karaoke-clipper",
        description="Detect sung-song regions and export clips from YouTube VOD or local MP4.",
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--url", help="YouTube VOD URL")
    source_group.add_argument("--file", help="Local MP4 file path")

    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="karaoke",
        help="Apply a preset tuning profile (use 'custom' to disable presets).",
    )

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
        "--merge-max-segment",
        type=float,
        default=None,
        help="Maximum length allowed when merging adjacent segments (defaults to --max-segment)",
    )
    parser.add_argument(
        "--segment-tolerance",
        type=float,
        default=0.0,
        help="Allow segment lengths to be +/- this many seconds when merging/splitting",
    )
    parser.add_argument(
        "--bridge-noise-gap",
        type=float,
        default=2.0,
        help="Bridge gaps labeled as noise up to this length (seconds)",
    )
    parser.add_argument(
        "--bridge-speech-gap",
        type=float,
        default=1.0,
        help="Bridge gaps labeled as speech up to this length (seconds)",
    )
    parser.add_argument(
        "--pre-roll",
        type=float,
        default=0.5,
        help="Padding seconds to add before each candidate clip",
    )
    parser.add_argument(
        "--post-roll",
        type=float,
        default=2.0,
        help="Padding seconds to add after each candidate clip",
    )
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
        "--whisperx-boundary-mode",
        choices=WHISPERX_BOUNDARY_MODES,
        default="safe",
        help="WhisperX boundary handling: off, metadata, or safe refinement",
    )
    parser.add_argument(
        "--whisperx-max-start-shrink",
        type=float,
        default=0.5,
        help="Maximum seconds WhisperX can trim from clip start in safe mode",
    )
    parser.add_argument(
        "--whisperx-max-end-shrink",
        type=float,
        default=0.5,
        help="Maximum seconds WhisperX can trim from clip end in safe mode",
    )
    parser.add_argument(
        "--allow-hard-split",
        type=str_to_bool,
        default=False,
        help="Allow hard splitting long segments at max length (true/false)",
    )
    parser.add_argument(
        "--energy-frame-ms",
        type=int,
        default=100,
        help="Energy fallback frame size in milliseconds",
    )
    parser.add_argument(
        "--energy-min-active-ms",
        type=int,
        default=500,
        help="Minimum active energy duration before starting a segment (ms)",
    )
    parser.add_argument(
        "--energy-min-silence-ms",
        type=int,
        default=1200,
        help="Minimum silence duration before closing a segment (ms)",
    )
    parser.add_argument(
        "--review-score-threshold",
        type=float,
        default=0.65,
        help="Score below which clips are marked for review",
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
        "--gdrive-upload-mode",
        choices=["clips", "all"],
        default="clips",
        help="Upload mode when gdrive upload is enabled: 'clips' or 'all'",
    )

    parser.add_argument(
        "--exclude-start-seconds",
        type=float,
        default=0.0,
        help="Seconds at start of audio to ignore for segmentation",
    )
    parser.add_argument(
        "--exclude-end-seconds",
        type=float,
        default=0.0,
        help="Seconds at end of audio to ignore for segmentation",
    )

    return parser


def load_config(argv: Optional[Sequence[str]] = None) -> AppConfig:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    argv_list = list(argv) if argv is not None else sys.argv[1:]

    def _flag_present(flag: str) -> bool:
        return flag in argv_list

    if args.profile != "custom" and args.profile in PROFILES:
        profile = PROFILES[args.profile]
        if not _flag_present("--min-segment"):
            args.min_segment = float(profile["min_segment"])
        if not _flag_present("--max-segment"):
            args.max_segment = float(profile["max_segment"])
        if not _flag_present("--merge-gap"):
            args.merge_gap = float(profile["merge_gap"])
        if not _flag_present("--bridge-noise-gap"):
            args.bridge_noise_gap = float(profile["bridge_noise_gap"])
        if not _flag_present("--bridge-speech-gap"):
            args.bridge_speech_gap = float(profile["bridge_speech_gap"])
        if not _flag_present("--pre-roll"):
            args.pre_roll = float(profile["pre_roll"])
        if not _flag_present("--post-roll"):
            args.post_roll = float(profile["post_roll"])
        if not _flag_present("--whisperx-boundary-mode"):
            args.whisperx_boundary_mode = str(profile["whisperx_boundary_mode"])

    if args.min_segment <= 0:
        parser.error("--min-segment must be > 0")
    if args.max_segment <= args.min_segment:
        parser.error("--max-segment must be greater than --min-segment")
    if args.merge_max_segment is not None and args.merge_max_segment <= 0:
        parser.error("--merge-max-segment must be > 0")
    if args.segment_tolerance < 0:
        parser.error("--segment-tolerance must be >= 0")
    if args.bridge_noise_gap < 0 or args.bridge_speech_gap < 0:
        parser.error("--bridge-noise-gap and --bridge-speech-gap must be >= 0")
    if args.pre_roll < 0 or args.post_roll < 0:
        parser.error("--pre-roll and --post-roll must be >= 0")
    if args.whisperx_max_start_shrink < 0 or args.whisperx_max_end_shrink < 0:
        parser.error("--whisperx-max-start-shrink and --whisperx-max-end-shrink must be >= 0")
    if args.energy_frame_ms <= 0:
        parser.error("--energy-frame-ms must be > 0")
    if args.energy_min_active_ms <= 0 or args.energy_min_silence_ms <= 0:
        parser.error("--energy-min-active-ms and --energy-min-silence-ms must be > 0")
    if args.review_score_threshold < 0 or args.review_score_threshold > 1:
        parser.error("--review-score-threshold must be between 0 and 1")
    if args.exclude_end_seconds is not None and args.exclude_end_seconds < 0:
        parser.error("--exclude-end-seconds must be >= 0")
    if args.expected_song_count is not None and args.expected_song_count <= 0:
        parser.error("--expected-song-count must be > 0")

    file_path = Path(args.file).expanduser().resolve() if args.file else None
    if file_path and not file_path.exists():
        parser.error(f"Input file does not exist: {file_path}")

    gdrive_folder_id = args.gdrive_folder_id or os.getenv("GDRIVE_FOLDER_ID")

    def _extract_drive_folder_id(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        val = str(value).strip()
        if "drive.google.com" in val:
            if "/folders/" in val:
                parts = val.split("/folders/")
                if len(parts) > 1:
                    return parts[1].split("?")[0].strip("/ ")
            if "id=" in val:
                for part in val.split("&"):
                    if part.startswith("id="):
                        return part.split("=", 1)[1]
        return val or None

    gdrive_folder_id = _extract_drive_folder_id(gdrive_folder_id)
    gdrive_client_secrets = (
        args.gdrive_client_secrets
        or os.getenv("GDRIVE_CLIENT_SECRETS")
        or "secret"
    )
    gdrive_token_value = os.getenv("GDRIVE_TOKEN") or args.gdrive_token

    merge_max_segment = args.merge_max_segment if args.merge_max_segment is not None else args.max_segment
    if merge_max_segment <= args.min_segment:
        parser.error("--merge-max-segment must be greater than --min-segment")

    effective_device = resolve_runtime_device(str(args.device))

    return AppConfig(
        url=args.url,
        file=file_path,
        outdir=Path(args.outdir).expanduser().resolve(),
        audio_clips=args.audio_clips,
        min_segment=float(args.min_segment),
        max_segment=float(merge_max_segment),
        use_acoustid=args.use_acoustid,
        ref_library=Path(args.ref_library).expanduser().resolve() if args.ref_library else None,
        device=effective_device,
        sample_rate=int(args.sample_rate),
        merge_gap=float(args.merge_gap),
        merge_max_segment=float(merge_max_segment),
        segment_tolerance=float(args.segment_tolerance),
        pre_roll_sec=float(args.pre_roll),
        post_roll_sec=float(args.post_roll),
        bridge_noise_gap_sec=float(args.bridge_noise_gap),
        bridge_speech_gap_sec=float(args.bridge_speech_gap),
        expected_song_count=args.expected_song_count,
        clip_mode=args.clip_mode,
        clip_resolution=args.clip_resolution,
        fingerprint_threshold=float(args.fingerprint_threshold),
        acoustid_api_key=os.getenv("ACOUSTID_API_KEY"),
        whisperx_boundary_mode=str(args.whisperx_boundary_mode),
        whisperx_max_start_shrink_sec=float(args.whisperx_max_start_shrink),
        whisperx_max_end_shrink_sec=float(args.whisperx_max_end_shrink),
        allow_hard_split=bool(args.allow_hard_split),
        energy_frame_ms=int(args.energy_frame_ms),
        energy_min_active_ms=int(args.energy_min_active_ms),
        energy_min_silence_ms=int(args.energy_min_silence_ms),
        profile=str(args.profile),
        review_score_threshold=float(args.review_score_threshold),
        gdrive_upload=bool(args.gdrive_upload),
        gdrive_folder_id=str(gdrive_folder_id).strip() if gdrive_folder_id else None,
        gdrive_client_secrets=(
            Path(gdrive_client_secrets).expanduser().resolve()
            if gdrive_client_secrets
            else None
        ),
        gdrive_token_path=Path(gdrive_token_value).expanduser().resolve(),
        gdrive_include_tmp=bool(args.gdrive_include_tmp),
        gdrive_upload_mode=str(args.gdrive_upload_mode),
        exclude_start_seconds=float(args.exclude_start_seconds),
        exclude_end_seconds=float(args.exclude_end_seconds),
    )
