"""Music candidate segmentation stage."""

from __future__ import annotations

import logging
import statistics
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Segment:
    start: float
    end: float
    label: str = "music"
    score: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def merge_adjacent_segments(
    segments: List[Segment],
    max_gap_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
) -> List[Segment]:
    """Merge nearby segments and enforce min/max duration limits."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    merged: List[Segment] = [Segment(ordered[0].start, ordered[0].end, ordered[0].label, ordered[0].score)]

    for seg in ordered[1:]:
        current = merged[-1]
        gap = seg.start - current.end
        if gap <= max_gap_sec:
            current.end = max(current.end, seg.end)
            current.score = max(current.score, seg.score)
        else:
            merged.append(Segment(seg.start, seg.end, seg.label, seg.score))

    clipped: List[Segment] = []
    for seg in merged:
        if seg.duration < min_segment_sec:
            continue

        if seg.duration <= max_segment_sec:
            clipped.append(seg)
            continue

        cursor = seg.start
        while cursor < seg.end:
            next_end = min(cursor + max_segment_sec, seg.end)
            chunk = Segment(cursor, next_end, seg.label, seg.score)
            if chunk.duration >= min_segment_sec:
                clipped.append(chunk)
            cursor = next_end

    return clipped


def _segments_with_ina(audio_path: Path) -> List[Segment]:
    from inaSpeechSegmenter import Segmenter  # type: ignore

    segmenter = Segmenter(detect_gender=False)
    raw = segmenter(str(audio_path))

    segments: List[Segment] = []
    for label, start, end in raw:
        lowered = str(label).lower()
        if "music" in lowered:
            segments.append(Segment(float(start), float(end), label="music", score=0.8))

    return segments


def _segments_with_energy_fallback(audio_path: Path, logger: logging.Logger) -> List[Segment]:
    """Fallback segmentation when inaSpeechSegmenter is unavailable."""
    logger.warning("Using fallback energy-based segmentation. Install inaSpeechSegmenter for better results.")

    with wave.open(str(audio_path), "rb") as wav_handle:
        sample_rate = wav_handle.getframerate()
        n_channels = wav_handle.getnchannels()
        sample_width = wav_handle.getsampwidth()
        frame_count = wav_handle.getnframes()
        payload = wav_handle.readframes(frame_count)

    if sample_width != 2:
        raise RuntimeError("Fallback segmentation expects 16-bit PCM WAV input")

    samples = array("h")
    samples.frombytes(payload)

    if n_channels > 1:
        mono = array("h")
        for idx in range(0, len(samples), n_channels):
            channel_frame = samples[idx : idx + n_channels]
            mono.append(int(sum(channel_frame) / len(channel_frame)))
        samples = mono

    window_size = max(sample_rate, 1)
    energies: List[float] = []

    for idx in range(0, len(samples), window_size):
        chunk = samples[idx : idx + window_size]
        if not chunk:
            continue
        energies.append(sum(abs(value) for value in chunk) / len(chunk))

    if not energies:
        return []

    threshold = max(250.0, statistics.median(energies) * 1.35)
    segments: List[Segment] = []

    active_start = None
    for window_idx, energy in enumerate(energies):
        if energy >= threshold and active_start is None:
            active_start = float(window_idx)
        elif energy < threshold and active_start is not None:
            segments.append(Segment(active_start, float(window_idx), label="music", score=0.35))
            active_start = None

    if active_start is not None:
        segments.append(Segment(active_start, float(len(energies)), label="music", score=0.35))

    return segments


def detect_music_segments(
    audio_path: Path,
    min_segment_sec: float,
    max_segment_sec: float,
    merge_gap_sec: float,
    logger: logging.Logger,
) -> List[Segment]:
    """Detect candidate music regions from the working WAV file."""
    base_segments: List[Segment]

    try:
        base_segments = _segments_with_ina(audio_path)
        logger.info("inaSpeechSegmenter produced %s raw music regions", len(base_segments))
    except Exception as exc:  # pragma: no cover - depends on optional package
        logger.warning("inaSpeechSegmenter unavailable or failed: %s", exc)
        base_segments = _segments_with_energy_fallback(audio_path, logger)

    merged = merge_adjacent_segments(
        base_segments,
        max_gap_sec=merge_gap_sec,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
    )
    logger.info("Segmentation finalized %s candidate regions", len(merged))
    return merged
