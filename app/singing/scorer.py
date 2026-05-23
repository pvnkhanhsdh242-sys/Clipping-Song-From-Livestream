"""Runtime scoring for trainable singing candidate models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from app.singing.features import FEATURE_NAMES, extract_candidate_features, vectorize_features


@dataclass(frozen=True)
class SingingScoreResult:
    score: float | None
    model_name: str
    decision: str


class SingingCandidateScorer:
    """Load and apply a singing candidate model artifact."""

    def __init__(
        self,
        *,
        model_path: Path | None,
        threshold: float,
        mode: str,
        logger,
    ) -> None:
        self.threshold = float(threshold)
        self.mode = str(mode)
        self.logger = logger
        self.model = None
        self.model_name = "none"
        self.backend = "none"
        self.feature_names = list(FEATURE_NAMES)

        if self.mode == "off":
            return
        if model_path is None:
            self.logger.warning("Singing model mode is %s but no model path was provided; scoring disabled.", self.mode)
            self.model_name = "missing"
            return

        self._load(model_path)

    @classmethod
    def from_config(cls, config, logger) -> "SingingCandidateScorer":
        return cls(
            model_path=getattr(config, "singing_model_path", None),
            threshold=float(getattr(config, "singing_score_threshold", 0.5)),
            mode=str(getattr(config, "singing_model_mode", "off")),
            logger=logger,
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and self.model is not None

    def _load(self, model_path: Path) -> None:
        artifact_path = model_path.expanduser().resolve()
        metadata_file = artifact_path / "metadata.json" if artifact_path.is_dir() else artifact_path.with_suffix(".metadata.json")
        metadata: dict[str, object] = {}
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                metadata = loaded

        backend = str(metadata.get("backend") or "").lower()
        model_type = str(metadata.get("model_type") or "")
        if backend == "pytorch" or model_type.startswith("pytorch") or artifact_path.suffix.lower() == ".pt":
            self._load_pytorch(artifact_path, metadata)
            return

        if artifact_path.is_dir():
            model_file_name = str(metadata.get("model_file") or "model.joblib")
            model_file = artifact_path / model_file_name
            if not model_file.exists() and (artifact_path / "model.pt").exists() and not (artifact_path / "model.joblib").exists():
                self._load_pytorch(artifact_path, metadata)
                return
            if not model_file.exists():
                model_file = artifact_path / "model.joblib"
        else:
            model_file = artifact_path

        self._load_sklearn(model_file, metadata)

    def _load_sklearn(self, model_file: Path, metadata: dict[str, object]) -> None:
        if not model_file.exists():
            self.logger.warning("Singing model file not found: %s; scoring disabled.", model_file)
            self.model_name = "missing"
            return

        try:
            import joblib  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
            self.logger.warning("Singing model scoring requires joblib: %s", exc)
            self.model_name = "dependency_missing"
            return

        self.model = joblib.load(model_file)
        feature_names = metadata.get("feature_names")
        if isinstance(feature_names, list) and all(isinstance(name, str) for name in feature_names):
            self.feature_names = list(feature_names)
        self.backend = "sklearn"
        self.model_name = str(metadata.get("model_type") or model_file.stem)
        self.logger.info("Loaded singing candidate model %s from %s", self.model_name, model_file)

    def _load_pytorch(self, artifact_path: Path, metadata: dict[str, object]) -> None:
        if artifact_path.is_dir():
            model_file_name = str(metadata.get("model_file") or "model.pt")
            model_file = artifact_path / model_file_name
            if not model_file.exists():
                model_file = artifact_path / "model.pt"
        else:
            model_file = artifact_path

        if not model_file.exists():
            self.logger.warning("Singing PyTorch model file not found: %s; scoring disabled.", model_file)
            self.model_name = "missing"
            return

        try:
            from app.singing.pytorch_backend import PyTorchCandidateModel

            self.model = PyTorchCandidateModel.load(model_file, metadata, device="auto")
        except Exception as exc:  # pragma: no cover - dependency/runtime dependent
            self.logger.warning("Unable to load PyTorch singing model %s: %s", model_file, exc)
            self.model_name = "dependency_missing"
            return

        self.backend = "pytorch"
        self.model_name = getattr(self.model, "model_name", str(metadata.get("model_type") or model_file.stem))
        self.logger.info("Loaded singing candidate model %s from %s", self.model_name, model_file)

    def score_features(self, features: Mapping[str, float]) -> SingingScoreResult:
        if self.mode == "off":
            return SingingScoreResult(score=None, model_name="none", decision="disabled")
        if self.model is None:
            return SingingScoreResult(score=None, model_name=self.model_name, decision="unavailable")
        if self.backend == "pytorch":
            return SingingScoreResult(score=None, model_name=self.model_name, decision="requires_audio")

        vector = [vectorize_features(features, self.feature_names)]
        score = self._predict_probability(vector)
        decision = "pass" if score >= self.threshold else "below_threshold"
        return SingingScoreResult(score=score, model_name=self.model_name, decision=decision)

    def score_candidate(
        self,
        audio_path: Path,
        start_sec: float,
        end_sec: float,
        *,
        music_ratio: float,
        fingerprint_confidence: float,
        duration_score: float,
        boundary_quality_score: float,
        merge_count: int,
        bridged_gap_total_sec: float,
        boundary_method: str,
    ) -> SingingScoreResult:
        if self.mode == "off":
            return SingingScoreResult(score=None, model_name="none", decision="disabled")
        if self.model is None:
            return SingingScoreResult(score=None, model_name=self.model_name, decision="unavailable")

        if self.backend == "pytorch":
            try:
                score = float(self.model.score_candidate(audio_path, start_sec, end_sec))
            except Exception as exc:
                self.logger.warning("Singing PyTorch scoring failed for %.3f -> %.3f: %s", start_sec, end_sec, exc)
                return SingingScoreResult(score=None, model_name=self.model_name, decision="feature_error")
            decision = "pass" if score >= self.threshold else "below_threshold"
            return SingingScoreResult(score=score, model_name=self.model_name, decision=decision)

        try:
            features = extract_candidate_features(
                audio_path,
                start_sec,
                end_sec,
                music_ratio=music_ratio,
                fingerprint_confidence=fingerprint_confidence,
                duration_score=duration_score,
                boundary_quality_score=boundary_quality_score,
                merge_count=merge_count,
                bridged_gap_total_sec=bridged_gap_total_sec,
                boundary_method=boundary_method,
            )
        except Exception as exc:
            self.logger.warning("Singing feature extraction failed for %.3f -> %.3f: %s", start_sec, end_sec, exc)
            return SingingScoreResult(score=None, model_name=self.model_name, decision="feature_error")

        return self.score_features(features)

    def _predict_probability(self, vector: list[list[float]]) -> float:
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            proba = predict_proba(vector)
            return _clamp_probability(self._positive_probability(proba[0]))

        decision_function = getattr(self.model, "decision_function", None)
        if callable(decision_function):
            raw = float(decision_function(vector)[0])
            return _clamp_probability(1.0 / (1.0 + math.exp(-raw)))

        predict = getattr(self.model, "predict", None)
        if callable(predict):
            return _clamp_probability(float(predict(vector)[0]))

        raise RuntimeError("Singing model does not expose predict_proba, decision_function, or predict.")

    def _positive_probability(self, probabilities) -> float:
        classes = getattr(self.model, "classes_", None)
        if classes is not None:
            class_values = [int(value) if str(value).isdigit() else value for value in list(classes)]
            if 1 in class_values:
                return float(probabilities[class_values.index(1)])
            if len(class_values) == 1 and class_values[0] == 0:
                return 0.0
            if len(class_values) == 1 and class_values[0] == 1:
                return 1.0
        return float(probabilities[-1])


def _clamp_probability(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))
