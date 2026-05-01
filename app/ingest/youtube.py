"""YouTube and local-source ingest logic."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.utils.ffmpeg import has_video_stream


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


def probe_youtube_metadata(url: str) -> tuple[str, str]:
    """Fetch YouTube metadata (title + id) without downloading."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Install dependencies first.") from exc

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if info is None:
        raise RuntimeError("yt-dlp returned no metadata")

    video_id = str(info.get("id") or "").strip()
    title = str(info.get("title") or "").strip()

    if not title:
        title = video_id or "unknown"
    if not video_id:
        video_id = title or "unknown"

    return video_id, title


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


def _find_existing_video_by_id(video_id: str, output_dir: Path) -> Optional[Path]:
    if not video_id:
        return None

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
    return candidates[0].resolve() if candidates else None


def _wait_for_partials(video_id: str, output_dir: Path, logger: logging.Logger, max_wait_sec: int = 300) -> None:
    if not video_id:
        return

    start = time.monotonic()
    while True:
        part_files = [
            path
            for path in output_dir.iterdir()
            if path.is_file() and video_id in path.name and path.name.endswith(".part")
        ]
        if not part_files:
            return

        if time.monotonic() - start > max_wait_sec:
            logger.warning("Timed out waiting for partial downloads to finish: %s", output_dir)
            return

        logger.info("Waiting for in-progress download to finish (%s partial files detected)", len(part_files))
        time.sleep(2)


def _acquire_download_lock(
    output_dir: Path,
    logger: logging.Logger,
    max_wait_sec: int = 1200,
) -> Path:
    lock_path = output_dir / ".download.lock"
    start = time.monotonic()

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(f"pid={os.getpid()}\n")
            return lock_path
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > max_wait_sec * 2:
                    logger.warning("Stale download lock detected; removing %s", lock_path)
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue

            if time.monotonic() - start > max_wait_sec:
                raise RuntimeError("Timed out waiting for download lock")

            time.sleep(2)


def download_youtube_video(url: str, output_dir: Path, logger: logging.Logger, retries: int = 3) -> SourceVideo:
    """Download a YouTube VOD with retry-safe behavior."""
    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is not installed. Install dependencies first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "%(title).120s_[%(id)s].%(ext)s"),
        "noplaylist": True,
        "writeinfojson": True,
        "continuedl": True,
        "retries": 3,
        "fragment_retries": 3,
        "file_access_retries": 20,
        "ignoreerrors": False,
        "quiet": True,
        "no_warnings": True,
    }

    last_error: Optional[Exception] = None

    lock_path = _acquire_download_lock(output_dir, logger)
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_probe = ydl.extract_info(url, download=False)
            if info_probe is None:
                raise RuntimeError("yt-dlp returned no metadata")

            video_id = str(info_probe.get("id") or "")
            title = str(info_probe.get("title") or "")
            existing = _find_existing_video_by_id(video_id, output_dir)
            if existing:
                info_json_path = _resolve_info_json(info_probe, output_dir, existing)
                if has_video_stream(existing, logger):
                    logger.info("Reusing existing download: %s", existing)
                    return SourceVideo(
                        source_mode="url",
                        video_id=video_id or existing.stem,
                        title=title or existing.stem,
                        video_path=existing,
                        metadata_path=info_json_path,
                    )
                logger.warning("Existing download has no video stream; re-downloading: %s", existing)

        _wait_for_partials(video_id, output_dir, logger)

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
                if not has_video_stream(video_path, logger):
                    logger.warning("Downloaded media has no video stream: %s", video_path)
                logger.info("Downloaded source video: %s", video_path)
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
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to remove download lock: %s", lock_path)

    raise RuntimeError(f"Failed to download URL after {retries} attempts") from last_error
