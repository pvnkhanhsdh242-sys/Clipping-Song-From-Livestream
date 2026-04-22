from pathlib import Path

from app.output.manifest import ManifestRecord, write_manifests


def test_write_manifests(tmp_path: Path):
    records = [
        ManifestRecord(
            source_video="sample.mp4",
            video_id="abc123",
            song="Song",
            artist="Artist",
            start_sec=10.2,
            end_sec=22.5,
            confidence=0.88,
            clip_path="output/clips/song_001.mp4",
            audio_path="output/clips/song_001.wav",
            backend="local-chromaprint",
        )
    ]

    json_path, csv_path = write_manifests(records, tmp_path / "manifests" / "abc123_manifest")

    assert json_path.exists()
    assert csv_path.exists()

    json_text = json_path.read_text(encoding="utf-8")
    assert "start_tc" in json_text
    assert "backend" in json_text
