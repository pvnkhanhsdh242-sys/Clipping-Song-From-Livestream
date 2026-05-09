"""Google Drive upload helpers (OAuth user flow)."""

from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path
from typing import Iterable, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


SCOPES = ["https://www.googleapis.com/auth/drive"]
CLIENT_SECRET_PATTERNS = [
    "client_secret*.json",
    "*apps.googleusercontent.com.json",
    "*client_secret*.json",
]


def _find_secrets_in_dir(search_dir: Path) -> Optional[Path]:
    for pattern in CLIENT_SECRET_PATTERNS:
        matches = sorted(search_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_client_secrets_path(explicit: Optional[Path]) -> Optional[Path]:
    if explicit:
        if explicit.is_dir():
            candidate = _find_secrets_in_dir(explicit)
            if candidate:
                return candidate
        elif explicit.exists():
            return explicit

    secret_dir = Path("secret")
    if secret_dir.exists():
        return _find_secrets_in_dir(secret_dir)

    return None


def _load_credentials(client_secrets_path: Path, token_path: Path, logger: logging.Logger) -> Credentials:
    token_path.parent.mkdir(parents=True, exist_ok=True)

    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing Google Drive token")
            creds.refresh(Request())
        else:
            logger.info("Starting Google Drive OAuth flow")
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.write_text(creds.to_json(), encoding="utf-8")

    return creds


def get_drive_service(
    client_secrets_path: Optional[Path],
    token_path: Path,
    logger: logging.Logger,
):
    secrets_path = find_client_secrets_path(client_secrets_path)
    if secrets_path is None:
        raise FileNotFoundError(
            "Google Drive client secrets JSON not found. "
            "Provide --gdrive-client-secrets or place a client_secret*.json in secret/."
        )

    creds = _load_credentials(secrets_path, token_path, logger)
    return build("drive", "v3", credentials=creds)


def ensure_drive_folder(service, name: str, parent_id: Optional[str], logger: logging.Logger) -> str:
    safe_name = name.strip() or "output"
    escaped_name = safe_name.replace("'", "\\'")
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{escaped_name}' "
        "and trashed=false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"

    response = service.files().list(q=query, spaces="drive", fields="files(id, name)").execute()
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {"name": safe_name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        metadata["parents"] = [parent_id]

    created = service.files().create(body=metadata, fields="id").execute()
    folder_id = created.get("id")
    logger.info("Created Drive folder '%s' (id=%s)", safe_name, folder_id)
    return folder_id


def upload_file(service, file_path: Path, parent_id: str, logger: logging.Logger) -> None:
    mime_type, _ = mimetypes.guess_type(str(file_path))
    media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)
    metadata = {"name": file_path.name, "parents": [parent_id]}
    service.files().create(body=metadata, media_body=media, fields="id").execute()
    logger.info("Uploaded %s", file_path.name)


def upload_directory(
    service,
    local_dir: Path,
    parent_id: str,
    include_tmp: bool,
    logger: logging.Logger,
) -> None:
    exclude = set()
    if not include_tmp:
        exclude.add("tmp")

    folder_map = {local_dir.resolve(): parent_id}

    for root, dirnames, filenames in os.walk(local_dir):
        root_path = Path(root).resolve()
        dirnames[:] = [
            name for name in dirnames if name not in exclude and not name.startswith(".")
        ]

        root_id = folder_map[root_path]
        for dirname in dirnames:
            child_path = (root_path / dirname).resolve()
            child_id = ensure_drive_folder(service, dirname, root_id, logger)
            folder_map[child_path] = child_id

        for filename in filenames:
            if filename.startswith("."):
                continue
            upload_file(service, root_path / filename, root_id, logger)


def upload_output_dir(
    output_dir: Path,
    parent_folder_id: str,
    client_secrets_path: Optional[Path],
    token_path: Path,
    include_tmp: bool,
    logger: logging.Logger,
) -> str:
    service = get_drive_service(client_secrets_path, token_path, logger)
    run_folder_id = ensure_drive_folder(service, output_dir.name, parent_folder_id, logger)
    upload_directory(service, output_dir, run_folder_id, include_tmp, logger)
    return run_folder_id


def upload_clips_dir(
    output_dir: Path,
    parent_folder_id: str,
    client_secrets_path: Optional[Path],
    token_path: Path,
    logger: logging.Logger,
    clip_files: Optional[Iterable[Path]] = None,
) -> str:
    """Upload only the `clips/` folder under a run output to the given Drive folder.

    Creates a run folder under the provided parent, then creates a `clips` child
    and uploads files from local `output_dir/clips` into it.
    Returns the Drive id of the created clips folder.
    """
    service = get_drive_service(client_secrets_path, token_path, logger)
    run_folder_id = ensure_drive_folder(service, output_dir.name, parent_folder_id, logger)

    clips_folder_id = ensure_drive_folder(service, "clips", run_folder_id, logger)

    local_clips = (output_dir / "clips").resolve()
    if clip_files is None:
        if not local_clips.exists():
            logger.warning("No clips folder to upload at %s", local_clips)
            return run_folder_id

        # upload files directly under clips (no subfolders by design)
        for item in sorted(local_clips.iterdir()):
            if item.is_file() and not item.name.startswith("."):
                upload_file(service, item, clips_folder_id, logger)
        return clips_folder_id

    unique_files = {Path(item).resolve() for item in clip_files}
    for item in sorted(unique_files):
        if not item.exists():
            logger.warning("Clip file missing for upload: %s", item)
            continue
        if item.is_dir() or item.name.startswith("."):
            continue
        if local_clips.exists() and local_clips not in item.parents:
            logger.warning("Skipping clip outside clips folder: %s", item)
            continue
        upload_file(service, item, clips_folder_id, logger)

    return clips_folder_id
