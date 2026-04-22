"""Build a local Chromaprint reference library from known songs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from app.identify.chromaprint_match import fingerprint_audio_file

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def iter_audio_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            yield path


def split_artist_title(stem: str) -> tuple[str, str]:
    if " - " in stem:
        artist, title = stem.split(" - ", 1)
        return artist.strip() or "Unknown Artist", title.strip() or stem
    return "Unknown Artist", stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Build local fingerprint library JSON")
    parser.add_argument("--input-dir", required=True, help="Folder with known song files")
    parser.add_argument("--output", default="data/reference_library.json", help="Output JSON path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_file = Path(args.output).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    tracks = []
    for idx, file_path in enumerate(iter_audio_files(input_dir), start=1):
        try:
            duration, fingerprint = fingerprint_audio_file(file_path)
        except Exception as exc:
            print(f"[WARN] Skip {file_path}: {exc}")
            continue

        artist, title = split_artist_title(file_path.stem)
        tracks.append(
            {
                "track_id": f"track-{idx:06d}",
                "title": title,
                "artist": artist,
                "source_path": str(file_path),
                "duration": round(duration, 3),
                "fingerprint": fingerprint,
            }
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "tracks": tracks,
    }

    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"Wrote {len(tracks)} tracks to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
