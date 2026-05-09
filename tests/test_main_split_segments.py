import logging

from app.main import SegmentAnalysis, _split_oversized_analyses


def test_split_oversized_analyses_chunks_by_max_segment():
    analyses = [
        SegmentAnalysis(
            index=1,
            start=549.7,
            end=1303.72,
            refined_start=549.7,
            refined_end=1303.72,
            match=None,
            confidence=0.5,
        )
    ]

    split = _split_oversized_analyses(analyses, max_segment_sec=360.0, logger=logging.getLogger("test"))

    assert len(split) == 3
    assert split[0].index == 1
    assert split[0].refined_start == 549.7
    assert split[0].refined_end == 909.7
    assert split[1].refined_start == 909.7
    assert split[1].refined_end == 1269.7
    assert split[2].refined_start == 1269.7
    assert split[2].refined_end == 1303.72
