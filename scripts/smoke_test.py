"""Quick smoke runner for one URL or local MP4."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run karaoke-clipper smoke test")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--file")
    parser.add_argument("--outdir", default="output/smoke")
    parser.add_argument(
        "--clip-resolution",
        choices=["source", "1080p", "720p", "480p", "360p"],
        default="source",
    )
    parser.add_argument("--expected-song-count", type=int, default=None)
    args = parser.parse_args()

    command = [
        sys.executable,
        "-m",
        "app.main",
        "--outdir",
        str(Path(args.outdir).expanduser()),
        "--audio-clips",
        "false",
        "--use-acoustid",
        "false",
        "--clip-resolution",
        args.clip_resolution,
    ]

    if args.url:
        command.extend(["--url", args.url])
    else:
        command.extend(["--file", args.file])

    if args.expected_song_count is not None:
        command.extend(["--expected-song-count", str(args.expected_song_count)])

    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
