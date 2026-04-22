from pathlib import Path

from app.ingest.youtube import _resolve_downloaded_video_path, _resolve_info_json


def test_resolve_downloaded_video_path_prefers_video_id(tmp_path: Path):
    wrong = tmp_path / "other_[xyz999].mp4"
    right = tmp_path / "title_[abc123].mp4"
    wrong.write_text("x", encoding="utf-8")
    right.write_text("y", encoding="utf-8")

    info = {"id": "abc123", "requested_downloads": []}
    resolved = _resolve_downloaded_video_path(info, tmp_path)
    assert resolved == right.resolve()


def test_resolve_info_json_from_video_id(tmp_path: Path):
    video = tmp_path / "title_[abc123].mp4"
    info_json = tmp_path / "title_[abc123].info.json"
    video.write_text("v", encoding="utf-8")
    info_json.write_text("{}", encoding="utf-8")

    info = {"id": "abc123"}
    resolved = _resolve_info_json(info, tmp_path, video)
    assert resolved == info_json.resolve()
