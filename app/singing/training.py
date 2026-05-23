"""Training helpers for the singing candidate scorer."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from app.singing.features import FEATURE_NAMES, extract_candidate_features, vectorize_features
from app.singing.labels import LabeledCandidate, load_labeled_candidates


AUDIO_SUFFIXES = {".wav"}


@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    model_path: Path
    metadata_path: Path
    labeled_count: int
    positive_count: int
    negative_count: int
    metrics: dict[str, float | int | str | None]


def _row_float(candidate: LabeledCandidate, key: str, default: float) -> float:
    if not candidate.row:
        return default
    value = candidate.row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def _extract_media_window(media_path: Path, start_sec: float, end_sec: float, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting {media_path}: {completed.stderr.strip()}")
    return output_wav


def _features_for_candidate(candidate: LabeledCandidate, tmp_dir: Path) -> dict[str, float]:
    source_path = candidate.source_video
    feature_audio = source_path
    feature_start = candidate.start_sec
    feature_end = candidate.end_sec

    if source_path.suffix.lower() not in AUDIO_SUFFIXES:
        feature_audio = tmp_dir / f"candidate_{candidate.row_index:06d}.wav"
        _extract_media_window(source_path, candidate.start_sec, candidate.end_sec, feature_audio)
        feature_start = 0.0
        feature_end = max(0.0, candidate.end_sec - candidate.start_sec)

    return extract_candidate_features(
        feature_audio,
        feature_start,
        feature_end,
        music_ratio=_row_float(candidate, "music_ratio", 0.0),
        fingerprint_confidence=_row_float(candidate, "fingerprint_confidence", 0.0),
        duration_score=_row_float(candidate, "duration_score", 0.0),
        boundary_quality_score=_row_float(candidate, "boundary_quality_score", 0.0),
        merge_count=int(_row_float(candidate, "merge_count", 0.0)),
        bridged_gap_total_sec=_row_float(candidate, "bridged_gap_total_sec", 0.0),
        boundary_method=str(candidate.row.get("boundary_method") if candidate.row else ""),
    )


def train_singing_candidate_model(
    manifest_paths: Sequence[Path],
    output_dir: Path,
    *,
    backend: str = "sklearn",
    validation_size: float = 0.25,
    random_state: int = 13,
    max_iter: int = 1000,
    device: str = "auto",
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    window_sec: float = 12.0,
    windows_per_clip: int = 4,
    logger: logging.Logger | None = None,
) -> TrainingResult:
    """Train and persist a singing candidate classifier artifact."""
    backend = str(backend or "sklearn").lower()
    if backend == "pytorch":
        from app.singing.pytorch_backend import train_pytorch_singing_candidate_model

        return train_pytorch_singing_candidate_model(
            manifest_paths,
            output_dir,
            validation_size=validation_size,
            random_state=random_state,
            epochs=max_iter,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            window_sec=window_sec,
            windows_per_clip=windows_per_clip,
            logger=logger,
            result_factory=TrainingResult,
        )
    if backend != "sklearn":
        raise ValueError("backend must be one of sklearn or pytorch")

    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
        from sklearn.dummy import DummyClassifier  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.metrics import accuracy_score, roc_auc_score  # type: ignore
        from sklearn.model_selection import train_test_split  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Training requires numpy, scikit-learn, and joblib.") from exc

    log = logger or logging.getLogger("karaoke_clipper.singing_training")
    candidates = load_labeled_candidates(manifest_paths)
    if not candidates:
        raise ValueError("No labeled rows found. Add label_singing to reviewed manifests first.")

    with tempfile.TemporaryDirectory(prefix="singing_training_") as tmp:
        tmp_dir = Path(tmp)
        feature_rows = [_features_for_candidate(candidate, tmp_dir) for candidate in candidates]

    x = np.array([vectorize_features(features, FEATURE_NAMES) for features in feature_rows], dtype=float)
    y = np.array([candidate.label_singing for candidate in candidates], dtype=int)
    positive_count = int(np.sum(y == 1))
    negative_count = int(np.sum(y == 0))

    metrics: dict[str, float | int | str | None] = {
        "labeled_count": int(y.size),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "validation_count": 0,
        "validation_accuracy": None,
        "validation_roc_auc": None,
    }

    unique_labels = set(int(value) for value in y)
    validation_count = int(math.ceil(float(y.size) * validation_size))
    train_count = int(y.size) - validation_count
    can_validate = (
        y.size >= 4
        and len(unique_labels) == 2
        and min(positive_count, negative_count) >= 2
        and validation_count >= 2
        and train_count >= 2
    )

    if len(unique_labels) < 2:
        log.warning("Only one label class was found; training a most-frequent dummy classifier.")
        model = DummyClassifier(strategy="most_frequent")
        model_type = "sklearn_dummy_classifier"
        model.fit(x, y)
    else:
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=random_state)),
            ]
        )
        model_type = "sklearn_logistic_regression"

        if can_validate:
            x_train, x_val, y_train, y_val = train_test_split(
                x,
                y,
                test_size=validation_size,
                random_state=random_state,
                stratify=y,
            )
            model.fit(x_train, y_train)
            predictions = model.predict(x_val)
            metrics["validation_count"] = int(y_val.size)
            metrics["validation_accuracy"] = float(accuracy_score(y_val, predictions))
            try:
                probabilities = model.predict_proba(x_val)[:, 1]
                metrics["validation_roc_auc"] = float(roc_auc_score(y_val, probabilities))
            except ValueError:
                metrics["validation_roc_auc"] = None
        else:
            log.warning("Too few labels for validation split; training on all labeled rows.")
            model.fit(x, y)

    if can_validate and len(unique_labels) == 2:
        final_model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=max_iter, class_weight="balanced", random_state=random_state)),
            ]
        )
        final_model.fit(x, y)
        model = final_model

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.joblib"
    metadata_path = output / "metadata.json"
    joblib.dump(model, model_path)

    metadata = {
        "schema_version": 1,
        "backend": "sklearn",
        "model_type": model_type,
        "model_file": "model.joblib",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": FEATURE_NAMES,
        "label_column": "label_singing",
        "threshold_default": 0.5,
        "training_epochs": int(max_iter),
        "metrics": metrics,
        "manifest_paths": [str(path.expanduser().resolve()) for path in manifest_paths],
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return TrainingResult(
        output_dir=output,
        model_path=model_path,
        metadata_path=metadata_path,
        labeled_count=int(y.size),
        positive_count=positive_count,
        negative_count=negative_count,
        metrics=metrics,
    )
