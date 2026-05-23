import csv
import json
import logging
import math
import struct
import wave
from pathlib import Path

import pytest

pytest.importorskip("torch")

from app.singing.pytorch_backend import PyTorchCandidateModel
from app.singing.scorer import SingingCandidateScorer
from app.singing.training import train_singing_candidate_model


def _write_tone(path: Path, *, frequency: float, amplitude: int = 12000, sample_rate: int = 16000) -> None:
    frames = []
    for idx in range(sample_rate):
        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * (idx / sample_rate)))
        frames.append(struct.pack("<h", sample))
    with wave.open(str(path), "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(b"".join(frames))


def _write_manifest(path: Path, rows: list[tuple[Path, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = ["source_video", "start_sec", "end_sec", "label_singing"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for source_video, label in rows:
            writer.writerow(
                {
                    "source_video": str(source_video),
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "label_singing": label,
                }
            )


def test_pytorch_training_creates_reloadable_artifact_and_scores_wav(tmp_path: Path):
    positive_a = tmp_path / "positive_a.wav"
    positive_b = tmp_path / "positive_b.wav"
    negative_a = tmp_path / "negative_a.wav"
    negative_b = tmp_path / "negative_b.wav"
    _write_tone(positive_a, frequency=440.0)
    _write_tone(positive_b, frequency=660.0)
    _write_tone(negative_a, frequency=90.0, amplitude=2500)
    _write_tone(negative_b, frequency=120.0, amplitude=2500)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(positive_a, 1), (negative_a, 0), (positive_b, 1), (negative_b, 0)])

    result = train_singing_candidate_model(
        [manifest],
        tmp_path / "artifact",
        backend="pytorch",
        max_iter=1,
        validation_size=0.5,
        device="cpu",
        batch_size=2,
        window_sec=0.5,
        windows_per_clip=1,
        logger=logging.getLogger("test"),
    )

    assert result.model_path.name == "model.pt"
    assert result.model_path.exists()
    assert result.metadata_path.exists()
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["backend"] == "pytorch"
    assert metadata["model_type"] == "pytorch_logspec_cnn"

    loaded = PyTorchCandidateModel.load(result.model_path, metadata, device="cpu")
    direct_score = loaded.score_candidate(positive_a, 0.0, 1.0)
    assert 0.0 <= direct_score <= 1.0

    scorer = SingingCandidateScorer(
        model_path=result.output_dir,
        threshold=0.5,
        mode="score",
        logger=logging.getLogger("test"),
    )
    score = scorer.score_candidate(
        positive_a,
        0.0,
        1.0,
        music_ratio=1.0,
        fingerprint_confidence=0.0,
        duration_score=0.4,
        boundary_quality_score=1.0,
        merge_count=0,
        bridged_gap_total_sec=0.0,
        boundary_method="test",
    )

    assert score.score is not None
    assert 0.0 <= score.score <= 1.0
    assert score.model_name == "pytorch_logspec_cnn"


def test_pytorch_scorer_requires_audio_for_feature_only_scoring(tmp_path: Path):
    positive = tmp_path / "positive.wav"
    negative = tmp_path / "negative.wav"
    _write_tone(positive, frequency=440.0)
    _write_tone(negative, frequency=110.0, amplitude=2500)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [(positive, 1), (negative, 0)])

    result = train_singing_candidate_model(
        [manifest],
        tmp_path / "artifact",
        backend="pytorch",
        max_iter=1,
        device="cpu",
        batch_size=2,
        window_sec=0.5,
        windows_per_clip=1,
        logger=logging.getLogger("test"),
    )
    scorer = SingingCandidateScorer(
        model_path=result.output_dir,
        threshold=0.5,
        mode="score",
        logger=logging.getLogger("test"),
    )

    result = scorer.score_features({})

    assert result.score is None
    assert result.decision == "requires_audio"
