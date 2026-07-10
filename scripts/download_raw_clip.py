"""Download only the raw VOD for a YouTube URL, using the standard output layout.

This reuses the same ingest logic and directory structure as the full pipeline:

    <outdir>/<sanitized title>/{vods,audio,clips,manifests,previews,logs,tmp}

The raw video lands in the ``vods`` subfolder and a mono working WAV is written
to the ``audio`` subfolder (same as the full pipeline). No segmentation,
recognition, or clip export is performed -- this just grabs the source VOD and
its audio track.

By default it grabs the highest-resolution H.264 (avc1) video + AAC (m4a) audio
muxed into MP4, which plays on every device with sound. yt-dlp's plain "best"
can pick AV1 video + Opus audio, which many Windows players cannot decode (you
get a broken-looking picture and no sound) -- avoid that with the default here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows consoles default to cp1252; VOD titles often contain CJK/Vietnamese
# text, so force UTF-8 to avoid UnicodeEncodeError when printing/logging.
for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="backslashreplace")

from app.ingest.youtube import (  # noqa: E402  (path bootstrap above)
    SourceVideo,
    download_youtube_video,
    probe_youtube_metadata,
)
from app.preprocess.extract_audio import extract_working_audio  # noqa: E402
from app.utils.logging import setup_logger  # noqa: E402
from app.utils.paths import prepare_output_dirs  # noqa: E402
from app.utils.timecode import sanitize_filename_component  # noqa: E402

DEFAULT_SAMPLE_RATE = 16000


# Highest-resolution H.264 video + AAC audio in MP4 -> max quality that still
# plays everywhere with sound. Falls back progressively if avc1/m4a is missing.
DEFAULT_FORMAT = (
    "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[ext=mp4]+bestaudio[ext=m4a]/"
    "best[ext=mp4]/best"
)

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".part"}


def _build_run_label(title: str, video_id: str) -> str:
    """Mirror app.main._build_run_label so folders match the full pipeline."""
    raw_label = title.strip() or video_id.strip() or "unknown"
    return sanitize_filename_component(raw_label)


def _remove_existing_downloads(video_id: str, vods_dir: Path, logger) -> None:
    """Drop any prior download for this id so a new format is fetched fresh."""
    if not video_id:
        return
    for path in vods_dir.iterdir():
        if not path.is_file() or video_id not in path.name:
            continue
        if path.suffix.lower() in VIDEO_SUFFIXES or path.name.endswith(".info.json"):
            logger.info("Removing existing download: %s", path)
            path.unlink(missing_ok=True)


def download_raw_clip(
    url: str,
    outdir: Path,
    format_selector: str = DEFAULT_FORMAT,
    overwrite: bool = False,
    extract_audio: bool = True,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> tuple[SourceVideo, Path | None]:
    """Download the raw VOD into ``vods`` and a working WAV into ``audio``.

    Returns the resolved source video plus the extracted WAV path (or ``None``
    when audio extraction is disabled).
    """
    video_id, title = probe_youtube_metadata(url)
    run_root = outdir / _build_run_label(title, video_id)
    output_dirs = prepare_output_dirs(run_root)

    logger = setup_logger(output_dirs["logs"] / "download_raw_clip.log")
    logger.info("Resolved run folder: %s", run_root)

    if overwrite:
        _remove_existing_downloads(video_id, output_dirs["vods"], logger)

    source = download_youtube_video(
        url,
        output_dirs["vods"],
        logger,
        format_selector=format_selector,
    )
    logger.info("Raw clip ready: %s", source.video_path)

    audio_path: Path | None = None
    if extract_audio:
        # Mirror app.main: <audio>/<video_id>.wav, mono PCM at the working rate.
        audio_path = output_dirs["audio"] / f"{source.video_id}.wav"
        extract_working_audio(source.video_path, audio_path, sample_rate, logger)

    return source, audio_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download only the raw YouTube VOD using the standard output layout.",
    )
    parser.add_argument("--url", required=True, help="YouTube VOD URL")
    parser.add_argument(
        "--outdir",
        default="output",
        help="Parent output directory (run subfolder named after source title)",
    )
    parser.add_argument(
        "--format",
        dest="format_selector",
        default=DEFAULT_FORMAT,
        help="yt-dlp format selector (default: max-res H.264 + AAC MP4)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download even if a file for this video id already exists",
    )
    parser.add_argument(
        "--no-audio",
        dest="extract_audio",
        action="store_false",
        help="Skip extracting the working WAV into the audio subfolder",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Working WAV sample rate (default: 16000)",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    source, audio_path = download_raw_clip(
        args.url,
        outdir,
        format_selector=args.format_selector,
        overwrite=args.overwrite,
        extract_audio=args.extract_audio,
        sample_rate=args.sample_rate,
    )

    print(f"Video ID : {source.video_id}")
    print(f"Title    : {source.title}")
    print(f"Raw clip : {source.video_path}")
    if source.metadata_path:
        print(f"Metadata : {source.metadata_path}")
    if audio_path:
        print(f"Audio WAV: {audio_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
