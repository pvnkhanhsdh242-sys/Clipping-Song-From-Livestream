"""Timecode helpers."""

from __future__ import annotations

import re

_TIMECODE_RE = re.compile(r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<ms>\d{1,3}))?$")


def seconds_to_timecode(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    if seconds < 0:
        raise ValueError("seconds must be >= 0")

    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def timecode_to_seconds(value: str) -> float:
    """Convert HH:MM:SS(.mmm) to seconds."""
    match = _TIMECODE_RE.match(value.strip())
    if not match:
        raise ValueError(f"Invalid timecode: {value}")

    hours = int(match.group("h"))
    minutes = int(match.group("m"))
    secs = int(match.group("s"))
    millis_str = match.group("ms") or "0"
    millis = int(millis_str.ljust(3, "0"))
    return hours * 3600 + minutes * 60 + secs + millis / 1000.0


def sanitize_filename_component(value: str) -> str:
    """Return a filesystem-safe label for clip names."""
    cleaned = re.sub(r"[^a-zA-Z0-9._ -]+", "_", value).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned or "unknown"
