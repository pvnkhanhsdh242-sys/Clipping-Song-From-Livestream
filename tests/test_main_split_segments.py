import logging

from app.main import SegmentAnalysis, _split_oversized_analyses


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
