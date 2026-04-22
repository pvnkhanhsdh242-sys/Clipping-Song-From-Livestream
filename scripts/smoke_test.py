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
    ]

    if args.url:
        command.extend(["--url", args.url])
    else:
        command.extend(["--file", args.file])

    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
