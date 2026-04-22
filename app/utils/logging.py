"""Logging configuration helpers."""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logger(log_file: Path, level: int = logging.INFO) -> logging.Logger:
    """Configure application logger for console + file output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("karaoke_clipper")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger
