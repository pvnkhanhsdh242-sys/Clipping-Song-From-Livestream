"""Audio extraction stage."""

from __future__ import annotations

import logging
from pathlib import Path

from app.utils.ffmpeg import run_command


def extract_working_audio(video_path: Path, output_wav: Path, sample_rate: int, logger: logging.Logger) -> Path:
    """Extract mono PCM WAV from input video."""
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    run_command(command, logger=logger)
    logger.info("Working audio ready: %s", output_wav)
    return output_wav
