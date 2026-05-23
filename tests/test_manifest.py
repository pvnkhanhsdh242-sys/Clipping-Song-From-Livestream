from pathlib import Path

from app.output.manifest import ManifestRecord, write_manifests


def test_write_manifests(tmp_path: Path):
    records = [
        ManifestRecord(
            source_video="sample.mp4",
            video_id="abc123",
            song="Song",
            artist="Artist",
            raw_start_sec=10.2,
            raw_end_sec=22.5,
            start_sec=10.2,
            end_sec=22.5,
            duration_sec=12.3,
            pre_roll_sec=0.5,
            post_roll_sec=2.0,
            boundary_method="ina",
            refinement_method="none",
            music_ratio=0.9,
            fingerprint_confidence=0.88,
            duration_score=1.0,
            boundary_quality_score=0.9,
            final_score=0.85,
            merge_count=0,
            bridged_gap_total_sec=0.0,
            needs_review=False,
            review_reason=None,
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
    assert "singing_score" in json_text
    assert "label_singing" in json_text
