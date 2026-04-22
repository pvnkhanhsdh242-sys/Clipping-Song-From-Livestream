"""Shared identification data types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchResult:
    """A best-match identification result for one segment."""

    song: str
    artist: str
    confidence: float
    backend: str
    track_id: Optional[str] = None
