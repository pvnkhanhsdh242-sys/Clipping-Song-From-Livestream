from app.segment.music_segments import Segment, merge_adjacent_segments


def test_merge_adjacent_segments_and_filter():
    segments = [
        Segment(0.0, 5.0, score=0.7),
        Segment(5.5, 12.0, score=0.8),
        Segment(20.0, 22.0, score=0.5),
    ]

    merged = merge_adjacent_segments(
        segments,
        max_gap_sec=1.0,
        min_segment_sec=4.0,
        max_segment_sec=30.0,
    )

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 12.0


def test_split_overly_long_segment():
    segments = [Segment(0.0, 70.0, score=0.9)]
    merged = merge_adjacent_segments(
        segments,
        max_gap_sec=1.0,
        min_segment_sec=10.0,
        max_segment_sec=30.0,
    )

    assert len(merged) == 3
    assert merged[0].duration == 30.0
    assert merged[1].duration == 30.0
    assert merged[2].duration == 10.0
