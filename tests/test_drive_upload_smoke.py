"""Smoke test for Google Drive upload.

Run with:

    RUN_GDRIVE_UPLOAD_SMOKE=1 pytest -q tests/test_drive_upload_smoke.py

Required environment variables:

    GDRIVE_FOLDER_ID

Optional environment variables:

    GDRIVE_CLIENT_SECRETS
    GDRIVE_TOKEN
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest


SMOKE_ENV_VAR = "RUN_GDRIVE_UPLOAD_SMOKE"


def _smoke_enabled() -> bool:
    return os.getenv(SMOKE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("gdrive_upload_smoke")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _resolve_optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _upload_hello_file(folder_id: str, client_secrets: Path | None, token_path: Path, file_path: Path) -> str:
    logger = _build_logger()
    try:
        from app.integrations.gdrive import get_drive_service
        from googleapiclient.http import MediaFileUpload
    except ModuleNotFoundError as exc:
        pytest.skip(f"Google Drive dependencies are not installed: {exc}")

    service = get_drive_service(client_secrets, token_path, logger)

    media = MediaFileUpload(str(file_path), mimetype="text/plain", resumable=True)
    created = service.files().create(
        body={"name": file_path.name, "parents": [folder_id]},
        media_body=media,
        fields="id, name",
    ).execute()

    file_id = created.get("id")
    assert file_id, "Drive upload did not return a file id"

    fetched = service.files().get(fileId=file_id, fields="id, name").execute()
    assert fetched.get("name") == file_path.name
    return str(file_id)


@pytest.mark.skipif(
    not _smoke_enabled(),
    reason=f"Set {SMOKE_ENV_VAR}=1 to run the Google Drive upload smoke test",
)
def test_upload_hello_txt_to_drive_smoke(tmp_path: Path):
    folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        pytest.skip("GDRIVE_FOLDER_ID is required for the Google Drive upload smoke test")

    client_secrets = _resolve_optional_path(os.getenv("GDRIVE_CLIENT_SECRETS"))
    token_path_value = os.getenv("GDRIVE_TOKEN", "secret/token.json")
    token_path = Path(token_path_value).expanduser().resolve()

    hello_path = tmp_path / "hello.txt"
    hello_path.write_text("hello from karaoke-clipper smoke test\n", encoding="utf-8")

    file_id = _upload_hello_file(folder_id, client_secrets, token_path, hello_path)
    assert file_id


def main() -> int:
    if not _smoke_enabled():
        print(f"Set {SMOKE_ENV_VAR}=1 to run this smoke test.")
        return 2

    folder_id = os.getenv("GDRIVE_FOLDER_ID", "").strip()
    if not folder_id:
        print("GDRIVE_FOLDER_ID is required.")
        return 2

    client_secrets = _resolve_optional_path(os.getenv("GDRIVE_CLIENT_SECRETS"))
    token_path_value = os.getenv("GDRIVE_TOKEN", "secret/token.json")
    token_path = Path(token_path_value).expanduser().resolve()

    hello_path = Path.cwd() / "hello.txt"
    hello_path.write_text("hello from karaoke-clipper smoke test\n", encoding="utf-8")

    try:
        file_id = _upload_hello_file(folder_id, client_secrets, token_path, hello_path)
        print(f"Uploaded {hello_path.name} to Drive. File id: {file_id}")
        return 0
    finally:
        try:
            hello_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())