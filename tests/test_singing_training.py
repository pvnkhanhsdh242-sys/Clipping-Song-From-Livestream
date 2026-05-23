import csv
import math
import logging
import struct
import wave
from pathlib import Path

from app.singing.scorer import SingingCandidateScorer
from app.singing.training import train_singing_candidate_model


def _write_training_wav(path: Path, sample_rate: int = 16000) -> None:
    frames = []
    for idx in range(sample_rate * 4):
        second = idx // sample_rate
        amplitude = 14000 if second in {0, 2} else 2500
        frequency = 440.0 if second in {0, 2} else 120.0
        sample = int(amplitude * math.sin(2.0 * math.pi * frequency * (idx / sample_rate)))
        frames.append(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(b"".join(frames))


def test_train_singing_candidate_model_creates_reloadable_artifact(tmp_path: Path):
    wav_path = tmp_path / "training.wav"
    _write_training_wav(wav_path)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "source_video",
            "start_sec",
            "end_sec",
            "label_singing",
            "music_ratio",
            "fingerprint_confidence",
            "duration_score",
            "boundary_quality_score",
            "merge_count",
            "bridged_gap_total_sec",
            "boundary_method",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, label in enumerate([1, 0, 1, 0]):
            writer.writerow(
                {
                    "source_video": str(wav_path),
                    "start_sec": idx,
                    "end_sec": idx + 1,
                    "label_singing": label,
                    "music_ratio": 0.9 if label else 0.2,
                    "fingerprint_confidence": 0.7 if label else 0.1,
                    "duration_score": 0.4,
                    "boundary_quality_score": 0.8,
                    "merge_count": 0,
                    "bridged_gap_total_sec": 0.0,
                    "boundary_method": "ina",
                }
            )

    result = train_singing_candidate_model(
        [manifest],
        tmp_path / "artifact",
        validation_size=0.5,
        logger=logging.getLogger("test"),
    )

    assert result.model_path.exists()
    assert result.metadata_path.exists()
    assert result.labeled_count == 4

    scorer = SingingCandidateScorer(
        model_path=result.output_dir,
        threshold=0.5,
        mode="score",
        logger=logging.getLogger("test"),
    )
    score = scorer.score_features({})
    assert score.score is not None
