"""Clip export stage using FFmpeg."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.utils.ffmpeg import run_command


@dataclass
class ClipExportResult:
    clip_path: Path
    audio_path: Optional[Path]


def _accurate_clip_command(video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(video_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(output_path),
    ]


def _fast_clip_command(video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(video_path),
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]


def export_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    clips_dir: Path,
    clip_stem: str,
    include_audio_clip: bool,
    mode: str,
    logger: logging.Logger,
) -> ClipExportResult:
    """Export MP4 clip and optional WAV clip for the selected interval."""
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_path = clips_dir / f"{clip_stem}.mp4"
    audio_path: Optional[Path] = clips_dir / f"{clip_stem}.wav" if include_audio_clip else None

    if mode == "fast":
        fast_result = run_command(_fast_clip_command(video_path, start_sec, end_sec, clip_path), logger, check=False)
        if fast_result.returncode != 0:
            logger.warning("Fast clip mode failed; retrying with accurate re-encode.")
            run_command(_accurate_clip_command(video_path, start_sec, end_sec, clip_path), logger)
    else:
        run_command(_accurate_clip_command(video_path, start_sec, end_sec, clip_path), logger)

    if audio_path:
        audio_command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-to",
            f"{end_sec:.3f}",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]
        run_command(audio_command, logger)

    return ClipExportResult(clip_path=clip_path, audio_path=audio_path)
