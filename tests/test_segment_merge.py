import logging

from app.segment.music_segments import Segment, coalesce_segments_to_expected_count, merge_adjacent_segments


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


def test_merge_adjacent_segments_skips_oversized_merge():
    segments = [
        Segment(0.0, 20.0, score=0.7),
        Segment(21.0, 45.0, score=0.8),
    ]

    merged = merge_adjacent_segments(
        segments,
        max_gap_sec=2.0,
        min_segment_sec=4.0,
        max_segment_sec=30.0,
    )

    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[0].end == 20.0
    assert merged[1].start == 21.0
    assert merged[1].end == 45.0


def test_merge_adjacent_segments_respects_merge_cap():
    segments = [
        Segment(0.0, 20.0, score=0.7),
        Segment(21.0, 45.0, score=0.8),
    ]

    merged = merge_adjacent_segments(
        segments,
        max_gap_sec=2.0,
        min_segment_sec=4.0,
        max_segment_sec=120.0,
        merge_max_segment_sec=30.0,
    )

    assert len(merged) == 2


def test_split_overly_long_segment():
    segments = [Segment(0.0, 70.0, score=0.9)]
    merged = merge_adjacent_segments(
        segments,
        max_gap_sec=1.0,
        min_segment_sec=10.0,
        max_segment_sec=30.0,
        allow_hard_split=True,
    )

    assert len(merged) == 3
    assert merged[0].duration == 30.0
    assert merged[1].duration == 30.0
    assert merged[2].duration == 10.0


def test_coalesce_to_expected_song_count_merges_smallest_gaps_first():
    segments = [
        Segment(0.0, 60.0, score=0.8),
        Segment(61.0, 120.0, score=0.8),
        Segment(121.5, 180.0, score=0.8),
    ]

    merged = coalesce_segments_to_expected_count(
        segments,
        expected_song_count=2,
        merge_gap_sec=2.0,
        max_segment_sec=240.0,
        logger=logging.getLogger("test"),
    )

    assert len(merged) == 2
    assert merged[0].start == 0.0
    assert merged[0].end == 120.0


def test_coalesce_to_expected_song_count_respects_large_pauses():
    segments = [
        Segment(0.0, 60.0, score=0.8),
        Segment(95.0, 120.0, score=0.8),
        Segment(121.0, 160.0, score=0.8),
    ]

    merged = coalesce_segments_to_expected_count(
        segments,
        expected_song_count=1,
        merge_gap_sec=2.0,
        max_segment_sec=240.0,
        logger=logging.getLogger("test"),
    )

    # One merge happens for the close pair, then large pause blocks further coalescing.
    assert len(merged) == 2


def test_coalesce_to_expected_song_count_adapts_gap_threshold():
    segments = [
        Segment(0.0, 20.0, score=0.8),
        Segment(33.0, 55.0, score=0.8),
        Segment(68.0, 90.0, score=0.8),
    ]

    merged = coalesce_segments_to_expected_count(
        segments,
        expected_song_count=1,
        merge_gap_sec=2.0,
        max_segment_sec=240.0,
        logger=logging.getLogger("test"),
    )

    assert len(merged) == 1
    assert merged[0].start == 0.0
    assert merged[0].end == 90.0


def test_coalesce_to_expected_song_count_skips_oversized_merge():
    segments = [
        Segment(0.0, 20.0, score=0.8),
        Segment(21.0, 45.0, score=0.8),
        Segment(46.0, 60.0, score=0.8),
    ]

    merged = coalesce_segments_to_expected_count(
        segments,
        expected_song_count=1,
        merge_gap_sec=2.0,
        max_segment_sec=30.0,
        logger=logging.getLogger("test"),
    )

    assert len(merged) == 3
