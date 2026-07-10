"""Download a YouTube video with both video + audio, merged into one MP4.

Usage:
    python download.py "https://www.youtube.com/watch?v=..."
    python download.py "<url>" -o my_folder
    python download.py "<url>" --quality 720

Requires: yt-dlp (Python package) and ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Windows consoles often default to cp1252, which cannot print non-ASCII titles
# (e.g. CJK/Vietnamese filenames). Force UTF-8 output so summaries never crash.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def build_format(quality: int | None) -> str:
    """Return a yt-dlp format string that always pairs video with audio.

    ``bestvideo*+bestaudio`` grabs the best video and best audio separately and
    lets ffmpeg mux them together, so the result always has sound. ``/best`` is a
    fallback for the rare case where only a pre-muxed stream is available.
    """
    if quality:
        return (
            f"bestvideo[height<={quality}]+bestaudio/"
            f"best[height<={quality}]/best"
        )
    return "bestvideo*+bestaudio/best"


def ffprobe_streams(path: Path) -> list[dict]:
    """Return the list of streams (video/audio) in a media file via ffprobe."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    try:
        return json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []


def verify_video_and_audio(path: Path) -> tuple[bool, bool]:
    """Return (has_video, has_audio) for a downloaded media file."""
    streams = ffprobe_streams(path)
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return has_video, has_audio


def download(url: str, output_dir: Path, quality: int | None) -> Path:
    try:
        import yt_dlp
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "yt-dlp is not installed. Run: pip install -r requirements.txt"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": build_format(quality),
        "merge_output_format": "mp4",
        "outtmpl": str(output_dir / "%(title).120s [%(id)s].%(ext)s"),
        "noplaylist": True,
        "writeinfojson": False,
        "retries": 5,
        "fragment_retries": 5,
        "continuedl": True,
        # Re-encode nothing; just remux into an mp4 container.
        "postprocessors": [
            {"key": "FFmpegVideoRemuxer", "preferedformat": "mp4"},
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        # Resolve the final file path after post-processing/merge.
        final_path = None
        for item in info.get("requested_downloads") or []:
            candidate = item.get("filepath")
            if candidate and Path(candidate).exists():
                final_path = Path(candidate)
                break
        if final_path is None:
            final_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

    return final_path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video with video + sound into one MP4."
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "-o", "--output", default="downloads",
        help="Output directory (default: ./downloads)",
    )
    parser.add_argument(
        "-q", "--quality", type=int, default=None,
        help="Max video height, e.g. 1080 or 720 (default: best available)",
    )
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("ERROR: ffmpeg not found on PATH. Install it first.", file=sys.stderr)
        return 2

    print(f"Downloading: {args.url}")
    path = download(args.url, Path(args.output), args.quality)

    has_video, has_audio = verify_video_and_audio(path)
    size_mb = path.stat().st_size / (1024 * 1024)

    print("\n--- Done ---")
    print(f"File:  {path}")
    print(f"Size:  {size_mb:.1f} MB")
    print(f"Video stream: {'YES' if has_video else 'NO'}")
    print(f"Audio stream: {'YES' if has_audio else 'NO'}")

    if not (has_video and has_audio):
        print("\nWARNING: file is missing a video or audio stream!", file=sys.stderr)
        return 1
    print("\nOK: file contains both video and sound.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
