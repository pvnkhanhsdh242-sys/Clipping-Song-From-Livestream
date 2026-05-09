import logging

from app.segment.music_segments import Segment, _apply_exclude_window


def test_apply_exclude_start_and_end():
    segments = [
        Segment(0.0, 5.0, score=0.7),
        Segment(10.0, 18.0, score=0.7),
    ]

    result = _apply_exclude_window(
        segments,
        exclude_start_seconds=2.0,
        exclude_end_seconds=3.0,
        audio_duration_sec=20.0,
        logger=logging.getLogger("test"),
    )

    assert len(result) == 2
    assert result[0].start == 2.0
    assert result[0].end == 5.0
    assert result[1].start == 10.0
    assert result[1].end == 17.0


def test_apply_exclude_end_drops_all():
    segments = [Segment(1.0, 4.0, score=0.7)]

    result = _apply_exclude_window(
        segments,
        exclude_start_seconds=0.0,
        exclude_end_seconds=15.0,
        audio_duration_sec=10.0,
        logger=logging.getLogger("test"),
    )

    assert result == []
