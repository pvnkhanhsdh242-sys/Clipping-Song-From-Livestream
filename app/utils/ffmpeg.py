"""FFmpeg-related command helpers."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class FFmpegError(RuntimeError):
    """Raised when an FFmpeg command fails."""


def ensure_ffmpeg_available() -> None:
    """Ensure ffmpeg and ffprobe are available on PATH."""
    missing = [binary for binary in ("ffmpeg", "ffprobe") if shutil.which(binary) is None]
    if missing:
        joined = ", ".join(missing)
        raise FFmpegError(f"Missing required binaries: {joined}")


def run_command(command: Sequence[str], logger: logging.Logger, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and log failures with stderr."""
    logger.debug("Running command: %s", " ".join(command))
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    if completed.returncode != 0:
        logger.error("Command failed (%s): %s", completed.returncode, " ".join(command))
        if completed.stderr:
            logger.error("stderr: %s", completed.stderr.strip())
        if check:
            raise FFmpegError(f"Command failed: {' '.join(command)}")

    return completed


def has_video_stream(media_path: Path, logger: logging.Logger) -> bool:
    """Return True if the media contains at least one video stream."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
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
        logger.warning("ffprobe failed for %s: %s", media_path, completed.stderr.strip())
        return False

    return "video" in completed.stdout.strip().lower()
