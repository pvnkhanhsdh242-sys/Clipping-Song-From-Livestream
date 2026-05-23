import math
import struct
import wave
from pathlib import Path

from app.singing.features import FEATURE_NAMES, extract_candidate_features, vectorize_features


def _write_sine_wav(path: Path, frequency: float = 440.0, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    frames = []
    for idx in range(int(sample_rate * duration_sec)):
        sample = int(12000 * math.sin(2.0 * math.pi * frequency * (idx / sample_rate)))
        frames.append(struct.pack("<h", sample))

    with wave.open(str(path), "wb") as wav_handle:
        wav_handle.setnchannels(1)
        wav_handle.setsampwidth(2)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(b"".join(frames))


def test_extract_candidate_features_from_wav(tmp_path: Path):
    wav_path = tmp_path / "tone.wav"
    _write_sine_wav(wav_path)

    features = extract_candidate_features(
        wav_path,
        0.0,
        1.0,
        music_ratio=0.8,
        fingerprint_confidence=0.7,
        duration_score=0.6,
        boundary_quality_score=0.9,
        merge_count=2,
        bridged_gap_total_sec=1.5,
        boundary_method="ina_bridge",
    )

    assert set(FEATURE_NAMES) == set(features)
    assert features["duration_sec"] == 1.0
    assert features["audio_rms"] > 0.0
    assert features["audio_peak"] > 0.0
    assert features["zero_crossing_rate"] > 0.0
    assert features["music_ratio"] == 0.8
    assert features["boundary_is_bridge"] == 1.0
    assert len(vectorize_features(features)) == len(FEATURE_NAMES)
