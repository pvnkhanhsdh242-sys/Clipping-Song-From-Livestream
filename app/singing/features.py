"""Audio and metadata feature extraction for singing candidate models."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Mapping


FEATURE_NAMES = [
    "duration_sec",
    "audio_rms",
    "audio_peak",
    "audio_silence_ratio",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "music_ratio",
    "fingerprint_confidence",
    "duration_score",
    "boundary_quality_score",
    "merge_count",
    "bridged_gap_total_sec",
    "boundary_is_energy_fallback",
    "boundary_is_bridge",
]


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Singing feature extraction requires numpy.") from exc
    return np


def _read_wav_window(audio_path: Path, start_sec: float, end_sec: float):
    np = _import_numpy()

    with wave.open(str(audio_path), "rb") as wav_handle:
        sample_rate = wav_handle.getframerate()
        channels = wav_handle.getnchannels()
        sample_width = wav_handle.getsampwidth()
        frame_count = wav_handle.getnframes()

        if sample_width != 2:
            raise RuntimeError("Singing feature extraction expects 16-bit PCM WAV input.")

        start_frame = max(0, min(frame_count, int(start_sec * sample_rate)))
        end_frame = max(start_frame, min(frame_count, int(end_sec * sample_rate)))
        wav_handle.setpos(start_frame)
        payload = wav_handle.readframes(end_frame - start_frame)

    samples = np.frombuffer(payload, dtype=np.int16)
    if samples.size == 0:
        return np.array([], dtype=np.float32), sample_rate

    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    return (samples.astype(np.float32) / 32768.0), sample_rate


def _spectral_features(samples, sample_rate: int) -> dict[str, float]:
    np = _import_numpy()
    if samples.size < 2 or sample_rate <= 0:
        return {
            "spectral_centroid_hz": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }

    max_samples = sample_rate * 30
    if samples.size > max_samples:
        indices = np.linspace(0, samples.size - 1, num=max_samples, dtype=np.int64)
        spectrum_input = samples[indices]
    else:
        spectrum_input = samples

    windowed = spectrum_input * np.hanning(spectrum_input.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / float(sample_rate))
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return {
            "spectral_centroid_hz": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }

    def _band_ratio(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs < high_hz)
        return float(np.sum(spectrum[mask]) / total)

    return {
        "spectral_centroid_hz": float(np.sum(freqs * spectrum) / total),
        "low_band_ratio": _band_ratio(80.0, 300.0),
        "mid_band_ratio": _band_ratio(300.0, 3000.0),
        "high_band_ratio": _band_ratio(3000.0, sample_rate / 2.0),
    }


def extract_candidate_features(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    music_ratio: float = 0.0,
    fingerprint_confidence: float = 0.0,
    duration_score: float = 0.0,
    boundary_quality_score: float = 0.0,
    merge_count: int = 0,
    bridged_gap_total_sec: float = 0.0,
    boundary_method: str = "",
) -> dict[str, float]:
    """Return a stable feature dict for one candidate segment."""
    np = _import_numpy()
    duration_sec = max(0.0, float(end_sec) - float(start_sec))
    samples, sample_rate = _read_wav_window(audio_path, start_sec, end_sec)

    if samples.size:
        abs_samples = np.abs(samples)
        audio_rms = float(np.sqrt(np.mean(np.square(samples))))
        audio_peak = float(np.max(abs_samples))
        audio_silence_ratio = float(np.mean(abs_samples < 0.01))
        zero_crossing_rate = float(np.mean(samples[:-1] * samples[1:] < 0.0)) if samples.size > 1 else 0.0
    else:
        audio_rms = 0.0
        audio_peak = 0.0
        audio_silence_ratio = 1.0
        zero_crossing_rate = 0.0

    boundary = boundary_method.lower()
    features = {
        "duration_sec": duration_sec,
        "audio_rms": audio_rms,
        "audio_peak": audio_peak,
        "audio_silence_ratio": audio_silence_ratio,
        "zero_crossing_rate": zero_crossing_rate,
        "music_ratio": float(music_ratio),
        "fingerprint_confidence": float(fingerprint_confidence),
        "duration_score": float(duration_score),
        "boundary_quality_score": float(boundary_quality_score),
        "merge_count": float(merge_count),
        "bridged_gap_total_sec": float(bridged_gap_total_sec),
        "boundary_is_energy_fallback": 1.0 if boundary.startswith("energy_fallback") else 0.0,
        "boundary_is_bridge": 1.0 if "bridge" in boundary else 0.0,
    }
    features.update(_spectral_features(samples, sample_rate))
    return {name: float(features.get(name, 0.0)) for name in FEATURE_NAMES}


def vectorize_features(features: Mapping[str, float], feature_names: list[str] | None = None) -> list[float]:
    """Convert a feature dict to the ordered vector expected by a model artifact."""
    names = feature_names or FEATURE_NAMES
    return [float(features.get(name, 0.0)) for name in names]
