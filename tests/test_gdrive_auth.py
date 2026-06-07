import logging
from pathlib import Path

import pytest

from app.integrations import gdrive


class _DummyRequest:
    def __init__(self, response: dict) -> None:
        self.response = response

    def execute(self) -> dict:
        return self.response


class _DummyFilesResource:
    def __init__(self, existing_files: list[dict] | None = None) -> None:
        self.existing_files = existing_files or []
        self.create_calls: list[dict] = []
        self.list_queries: list[str] = []

    def list(self, **kwargs):
        self.list_queries.append(str(kwargs.get("q", "")))
        return _DummyRequest({"files": self.existing_files})

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return _DummyRequest({"id": "new-file-id"})


class _DummyService:
    def __init__(self, files_resource: _DummyFilesResource) -> None:
        self.files_resource = files_resource

    def files(self):
        return self.files_resource


class _ExpiredCredentials:
    valid = False
    expired = True
    refresh_token = "stale-refresh-token"

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def refresh(self, request) -> None:
        raise self._exc


class _FreshCredentials:
    valid = True
    expired = False
    refresh_token = "fresh-refresh-token"

    def to_json(self) -> str:
        return '{"token": "fresh"}'


class _DummyFlow:
    def __init__(self) -> None:
        self.started = False

    def run_local_server(self, port: int = 0):
        self.started = True
        return _FreshCredentials()


def test_load_credentials_deletes_invalid_grant_token_and_reauths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "stale"}', encoding="utf-8")
    client_secret = tmp_path / "client_secret.json"
    client_secret.write_text("{}", encoding="utf-8")
    flow = _DummyFlow()

    monkeypatch.setattr(
        gdrive.Credentials,
        "from_authorized_user_file",
        lambda path, scopes: _ExpiredCredentials(
            gdrive.RefreshError("invalid_grant: Bad Request", {"error": "invalid_grant"})
        ),
    )
    monkeypatch.setattr(
        gdrive.InstalledAppFlow,
        "from_client_secrets_file",
        lambda path, scopes: flow,
    )

    creds = gdrive._load_credentials(client_secret, token_path, logging.getLogger("test_gdrive_auth"))

    assert isinstance(creds, _FreshCredentials)
    assert flow.started
    assert token_path.read_text(encoding="utf-8") == '{"token": "fresh"}'


def test_load_credentials_reraises_non_invalid_grant_refresh_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token": "stale"}', encoding="utf-8")
    client_secret = tmp_path / "client_secret.json"
    client_secret.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        gdrive.Credentials,
        "from_authorized_user_file",
        lambda path, scopes: _ExpiredCredentials(gdrive.RefreshError("temporarily unavailable")),
    )

    with pytest.raises(gdrive.RefreshError):
        gdrive._load_credentials(client_secret, token_path, logging.getLogger("test_gdrive_auth"))

    assert token_path.read_text(encoding="utf-8") == '{"token": "stale"}'


def test_upload_file_skips_existing_drive_file(tmp_path: Path):
    files_resource = _DummyFilesResource(existing_files=[{"id": "existing-id", "name": "clip.mp4"}])
    service = _DummyService(files_resource)
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"placeholder")

    gdrive.upload_file(service, clip_path, "parent-id", logging.getLogger("test_gdrive_auth"))

    assert files_resource.create_calls == []
    assert "name='clip.mp4'" in files_resource.list_queries[0]
    assert "'parent-id' in parents" in files_resource.list_queries[0]


def test_upload_file_creates_when_missing(tmp_path: Path):
    files_resource = _DummyFilesResource(existing_files=[])
    service = _DummyService(files_resource)
    clip_path = tmp_path / "clip.mp4"
    clip_path.write_bytes(b"placeholder")

    gdrive.upload_file(service, clip_path, "parent-id", logging.getLogger("test_gdrive_auth"))

    assert len(files_resource.create_calls) == 1
    assert files_resource.create_calls[0]["body"] == {"name": "clip.mp4", "parents": ["parent-id"]}
