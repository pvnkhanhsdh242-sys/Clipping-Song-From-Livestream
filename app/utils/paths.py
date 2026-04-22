"""Path and output layout utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


OUTPUT_SUBDIRS = {
    "vods": "vods",
    "audio": "audio",
    "clips": "clips",
    "manifests": "manifests",
    "logs": "logs",
    "tmp": "tmp",
}


def prepare_output_dirs(base: Path) -> Dict[str, Path]:
    """Create and return standardized output subdirectories."""
    base.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Path] = {}
    for key, relative in OUTPUT_SUBDIRS.items():
        target = base / relative
        target.mkdir(parents=True, exist_ok=True)
        result[key] = target
    return result
