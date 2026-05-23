import logging

from app.main import (
    SegmentAnalysis,
    _filter_analyses_by_music_ratio,
    _filter_analyses_by_singing_score,
    _split_oversized_analyses,
)


def _analysis(index: int, music_ratio: float) -> SegmentAnalysis:
    return SegmentAnalysis(
        index=index,
        raw_start=10.0 * index,
        raw_end=10.0 * index + 5.0,
        padded_start=10.0 * index,
        padded_end=10.0 * index + 5.0,
        refined_start=10.0 * index,
        refined_end=10.0 * index + 5.0,
        match=None,
        confidence=0.5,
        boundary_method="ina",
        refinement_method="none",
        music_ratio=music_ratio,
        fingerprint_confidence=0.0,
        duration_score=0.7,
        boundary_quality_score=0.9,
        final_score=0.6,
        merge_count=0,
        bridged_gap_total_sec=0.0,
        needs_review=False,
        review_reason=None,
    )


def test_split_oversized_analyses_chunks_by_max_segment():
    analyses = [
        SegmentAnalysis(
            index=1,
            raw_start=549.7,
            raw_end=1303.72,
            padded_start=549.7,
            padded_end=1303.72,
            refined_start=549.7,
            refined_end=1303.72,
            match=None,
            confidence=0.5,
            boundary_method="ina",
            refinement_method="none",
            music_ratio=0.9,
            fingerprint_confidence=0.0,
            duration_score=0.7,
            boundary_quality_score=0.9,
            final_score=0.6,
            merge_count=0,
            bridged_gap_total_sec=0.0,
            needs_review=False,
            review_reason=None,
        )
    ]

    split = _split_oversized_analyses(
        analyses,
        max_segment_sec=360.0,
        allow_hard_split=True,
        logger=logging.getLogger("test"),
    )

    assert len(split) == 3
    assert split[0].index == 1
    assert split[0].refined_start == 549.7
    assert split[0].refined_end == 909.7
    assert split[1].refined_start == 909.7
    assert split[1].refined_end == 1269.7
    assert split[2].refined_start == 1269.7
    assert split[2].refined_end == 1303.72


def test_filter_analyses_by_music_ratio_filters_and_reindexes():
    analyses = [
        _analysis(index=1, music_ratio=0.35),
        _analysis(index=2, music_ratio=0.61),
        _analysis(index=3, music_ratio=0.92),
    ]

    kept = _filter_analyses_by_music_ratio(
        analyses,
        music_ratio_threshold=0.6,
        logger=logging.getLogger("test"),
    )

    assert len(kept) == 2
    assert kept[0].music_ratio == 0.61
    assert kept[1].music_ratio == 0.92
    assert kept[0].index == 1
    assert kept[1].index == 2


def test_filter_analyses_by_music_ratio_disabled_when_zero():
    analyses = [
        _analysis(index=1, music_ratio=0.2),
        _analysis(index=2, music_ratio=0.8),
    ]

    kept = _filter_analyses_by_music_ratio(
        analyses,
        music_ratio_threshold=0.0,
        logger=logging.getLogger("test"),
    )

    assert len(kept) == 2
    assert kept[0].index == 1
    assert kept[1].index == 2


def test_filter_analyses_by_singing_score_filters_and_reindexes():
    analyses = [
        _analysis(index=1, music_ratio=0.8),
        _analysis(index=2, music_ratio=0.8),
        _analysis(index=3, music_ratio=0.8),
    ]
    analyses[0].singing_score = 0.2
    analyses[1].singing_score = 0.7
    analyses[2].singing_score = None

    kept = _filter_analyses_by_singing_score(
        analyses,
        singing_model_mode="filter",
        singing_score_threshold=0.5,
        logger=logging.getLogger("test"),
    )

    assert len(kept) == 2
    assert kept[0].singing_score == 0.7
    assert kept[1].singing_score is None
    assert kept[0].index == 1
    assert kept[1].index == 2


def test_filter_analyses_by_singing_score_disabled_in_score_mode():
    analyses = [_analysis(index=1, music_ratio=0.8)]
    analyses[0].singing_score = 0.1

    kept = _filter_analyses_by_singing_score(
        analyses,
        singing_model_mode="score",
        singing_score_threshold=0.5,
        logger=logging.getLogger("test"),
    )

    assert kept == analyses
