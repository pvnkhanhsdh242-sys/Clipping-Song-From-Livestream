"""Upload an existing pipeline output folder to Google Drive."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import str_to_bool
from app.integrations.gdrive import upload_clips_dir, upload_output_dir
from app.utils.logging import setup_logger


def _extract_drive_folder_id(value: str) -> str:
    val = value.strip()
    if "drive.google.com" in val:
        if "/folders/" in val:
            return val.split("/folders/", 1)[1].split("?")[0].strip("/ ")
        if "id=" in val:
            for part in val.split("&"):
                if part.startswith("id="):
                    return part.split("=", 1)[1]
    return val


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload an existing karaoke-clipper run folder to Google Drive.")
    parser.add_argument("--output-dir", required=True, help="Run output folder, e.g. output/<title>")
    parser.add_argument(
        "--parent-folder-id",
        default=os.getenv("GDRIVE_FOLDER_ID"),
        help="Destination Drive folder ID or URL. Defaults to GDRIVE_FOLDER_ID.",
    )
    parser.add_argument(
        "--client-secrets",
        default=os.getenv("GDRIVE_CLIENT_SECRETS") or "secret",
        help="Client secrets JSON path or directory. Defaults to secret.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GDRIVE_TOKEN") or "secret/token.json",
        help="OAuth token cache path.",
    )
    parser.add_argument("--mode", choices=["clips", "all"], default="clips", help="Upload clips only or all run files.")
    parser.add_argument("--include-tmp", type=str_to_bool, default=False, help="Include tmp folder for --mode all.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output folder does not exist: {output_dir}")
    if not args.parent_folder_id:
        raise ValueError("--parent-folder-id or GDRIVE_FOLDER_ID is required")

    parent_folder_id = _extract_drive_folder_id(str(args.parent_folder_id))
    client_secrets = Path(args.client_secrets).expanduser().resolve() if args.client_secrets else None
    token_path = Path(args.token).expanduser().resolve()
    logger = setup_logger(output_dir / "logs" / "gdrive_upload_manual.log", name="karaoke_clipper_gdrive_manual")

    if args.mode == "all":
        folder_id = upload_output_dir(
            output_dir=output_dir,
            parent_folder_id=parent_folder_id,
            client_secrets_path=client_secrets,
            token_path=token_path,
            include_tmp=bool(args.include_tmp),
            logger=logger,
        )
    else:
        folder_id = upload_clips_dir(
            output_dir=output_dir,
            parent_folder_id=parent_folder_id,
            client_secrets_path=client_secrets,
            token_path=token_path,
            logger=logger,
        )

    print(f"Uploaded to Google Drive folder id: {folder_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
