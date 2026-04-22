"""YouTube and local-source ingest logic."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class SourceVideo:
    """Represents the source video used by the pipeline."""

    source_mode: str
    video_id: str
    title: str
    video_path: Path
    metadata_path: Optional[Path] = None


def register_local_video(file_path: Path) -> SourceVideo:
    """Register a local MP4 as pipeline source."""
    resolved = file_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Local source does not exist: {resolved}")

    return SourceVideo(
        source_mode="local",
        video_id=resolved.stem,
        title=resolved.stem,
        video_path=resolved,
        metadata_path=None,
    )


def _resolve_downloaded_video_path(info: dict[str, Any], output_dir: Path) -> Path:
    requested = info.get("requested_downloads") or []
    for item in requested:
        filepath = item.get("filepath")
        if filepath:
            candidate = Path(filepath)
            if candidate.exists():
                return candidate.resolve()

    video_id = str(info.get("id") or "")
    if video_id:
        candidates = sorted(
            [
                path
                for path in output_dir.iterdir()
                if path.is_file()
                and video_id in path.name
                and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()

    raise FileNotFoundError("Could not locate downloaded video file")


def _resolve_info_json(info: dict[str, Any], output_dir: Path, video_path: Path) -> Optional[Path]:
    preferred = video_path.with_suffix(".info.json")
    if preferred.exists():
        return preferred.resolve()

    video_id = str(info.get("id") or "")
    if video_id:
        candidates = sorted(
            [
                path
                for path in output_dir.iterdir()
                if path.is_file() and video_id in path.name and path.name.endswith(".info.json")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0].resolve()

    return None


def download_youtube_video(url: str, output_dir: Path, logger: logging.Logger, retries: int = 3) -> SourceVideo:
    """Download a YouTube VOD with retry-safe behavior."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Install dependencies first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "%(title).120s_[%(id)s].%(ext)s"),
        "noplaylist": True,
        "writeinfojson": True,
        "continuedl": True,
        "retries": 3,
        "fragment_retries": 3,
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            logger.info("Downloading URL (attempt %s/%s): %s", attempt, retries, url)
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("yt-dlp returned no metadata")

            video_path = _resolve_downloaded_video_path(info, output_dir)
            info_json_path = _resolve_info_json(info, output_dir, video_path)

            video_id = str(info.get("id") or video_path.stem)
            title = str(info.get("title") or video_path.stem)
            logger.info("Downloaded source video: %s", video_path)

            return SourceVideo(
                source_mode="url",
                video_id=video_id,
                title=title,
                video_path=video_path,
                metadata_path=info_json_path,
            )
        except Exception as exc:  # pragma: no cover - network/tooling dependent path
            last_error = exc
            logger.warning("Download attempt %s failed: %s", attempt, exc)
            if attempt < retries:
                backoff = attempt * 2
                logger.info("Retrying in %s seconds", backoff)
                time.sleep(backoff)

    raise RuntimeError(f"Failed to download URL after {retries} attempts") from last_error
