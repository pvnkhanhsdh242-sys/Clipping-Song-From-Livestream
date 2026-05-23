import json
import logging
from pathlib import Path

import numpy as np

from app.singing.features import FEATURE_NAMES
from app.singing.scorer import SingingCandidateScorer


class FixedProbabilityModel:
    def __init__(self, score: float):
        self.score = score
        self.classes_ = np.array([0, 1])

    def predict_proba(self, rows):
        return np.array([[1.0 - self.score, self.score] for _ in rows])


class NegativeOnlyModel:
    classes_ = np.array([0])

    def predict_proba(self, rows):
        return np.array([[1.0] for _ in rows])


def _write_artifact(path: Path, score: float) -> Path:
    import joblib

    path.mkdir(parents=True)
    joblib.dump(FixedProbabilityModel(score), path / "model.joblib")
    (path / "metadata.json").write_text(
        json.dumps({"model_type": "fixed_test_model", "feature_names": FEATURE_NAMES}),
        encoding="utf-8",
    )
    return path


def test_singing_scorer_missing_model_is_unavailable(tmp_path: Path):
    scorer = SingingCandidateScorer(
        model_path=tmp_path / "missing",
        threshold=0.5,
        mode="score",
        logger=logging.getLogger("test"),
    )

    result = scorer.score_features({})

    assert result.score is None
    assert result.decision == "unavailable"


def test_singing_scorer_applies_threshold(tmp_path: Path):
    scorer = SingingCandidateScorer(
        model_path=_write_artifact(tmp_path / "model", 0.7),
        threshold=0.6,
        mode="score",
        logger=logging.getLogger("test"),
    )

    result = scorer.score_features({})

    assert result.score == 0.7
    assert result.model_name == "fixed_test_model"
    assert result.decision == "pass"


def test_singing_scorer_marks_below_threshold(tmp_path: Path):
    scorer = SingingCandidateScorer(
        model_path=_write_artifact(tmp_path / "model", 0.2),
        threshold=0.6,
        mode="filter",
        logger=logging.getLogger("test"),
    )

    result = scorer.score_features({})

    assert result.score == 0.2
    assert result.decision == "below_threshold"


def test_singing_scorer_handles_single_negative_class(tmp_path: Path):
    import joblib

    artifact = tmp_path / "model"
    artifact.mkdir()
    joblib.dump(NegativeOnlyModel(), artifact / "model.joblib")

    scorer = SingingCandidateScorer(
        model_path=artifact,
        threshold=0.6,
        mode="score",
        logger=logging.getLogger("test"),
    )

    result = scorer.score_features({})

    assert result.score == 0.0
    assert result.decision == "below_threshold"
