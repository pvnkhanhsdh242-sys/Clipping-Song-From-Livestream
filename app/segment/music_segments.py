"""Music candidate segmentation stage."""

from __future__ import annotations

import logging
import statistics
import sys
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
    merge_max_segment_sec: float | None = None,
    logger: logging.Logger | None = None,
) -> List[Segment]:
    """Merge nearby segments and enforce min/max duration limits."""
    if not segments:
        return []

    merge_cap_sec = merge_max_segment_sec if merge_max_segment_sec is not None else max_segment_sec

    ordered = sorted(segments, key=lambda s: s.start)
    merged: List[Segment] = [Segment(ordered[0].start, ordered[0].end, ordered[0].label, ordered[0].score)]

    for seg in ordered[1:]:
        current = merged[-1]
        gap = seg.start - current.end
        combined_duration = max(current.end, seg.end) - current.start
        if gap <= max_gap_sec and combined_duration <= merge_cap_sec:
            current.end = max(current.end, seg.end)
            current.score = max(current.score, seg.score)
        else:
            if gap <= max_gap_sec and combined_duration > merge_cap_sec and logger is not None:
                logger.info(
                    "Skipped merging adjacent segments at %.2fs-%.2fs and %.2fs-%.2fs: combined duration %.2fs would exceed merge cap %.2fs",
                    current.start,
                    current.end,
                    seg.start,
                    seg.end,
                    combined_duration,
                    merge_cap_sec,
                )
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


def coalesce_segments_to_expected_count(
    segments: List[Segment],
    expected_song_count: int,
    merge_gap_sec: float,
    max_segment_sec: float,
    logger: logging.Logger,
    merge_max_segment_sec: float | None = None,
) -> List[Segment]:
    """Best-effort merge of nearest neighbors until target song count is reached."""
    if expected_song_count <= 0 or len(segments) <= expected_song_count:
        return segments

    merge_cap_sec = merge_max_segment_sec if merge_max_segment_sec is not None else max_segment_sec

    coalesced = [Segment(s.start, s.end, s.label, s.score) for s in sorted(segments, key=lambda seg: seg.start)]
    original_count = len(coalesced)

    # Start from a conservative limit, then raise adaptively toward the user-provided target.
    base_limit_sec = max(12.0, merge_gap_sec * 6.0)
    hard_cap_sec = max(30.0, merge_gap_sec * 10.0)
    gap_values = [max(0.0, coalesced[idx + 1].start - coalesced[idx].end) for idx in range(len(coalesced) - 1)]
    required_merges = max(0, len(coalesced) - expected_song_count)
    adaptive_limit_sec = base_limit_sec
    if gap_values and required_merges > 0:
        sorted_gaps = sorted(gap_values)
        pivot = min(required_merges - 1, len(sorted_gaps) - 1)
        adaptive_limit_sec = max(base_limit_sec, sorted_gaps[pivot])

    max_bridge_gap_sec = min(adaptive_limit_sec, hard_cap_sec)
    logger.info(
        "Expected-song merge limits: base=%.2fs adaptive=%.2fs applied=%.2fs (cap=%.2fs)",
        base_limit_sec,
        adaptive_limit_sec,
        max_bridge_gap_sec,
        hard_cap_sec,
    )

    while len(coalesced) > expected_song_count:
        best_idx: int | None = None
        best_gap: float | None = None
        skipped_oversized = 0

        for idx in range(len(coalesced) - 1):
            gap = coalesced[idx + 1].start - coalesced[idx].end
            combined_duration = coalesced[idx + 1].end - coalesced[idx].start
            if combined_duration > merge_cap_sec:
                skipped_oversized += 1
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_idx = idx

        if best_idx is None or best_gap is None:
            if skipped_oversized:
                logger.warning(
                    "Expected-song merge stopped at %s segments (target=%s): all remaining candidate merges would exceed %.2fs",
                    len(coalesced),
                    expected_song_count,
                    merge_cap_sec,
                )
            else:
                logger.warning(
                    "Expected-song merge stopped at %s segments (target=%s): no merge candidates remain",
                    len(coalesced),
                    expected_song_count,
                )
            break

        if best_gap > max_bridge_gap_sec:
            logger.warning(
                "Expected-song merge stopped at %s segments (target=%s): nearest remaining gap %.2fs exceeds %.2fs",
                len(coalesced),
                expected_song_count,
                best_gap,
                max_bridge_gap_sec,
            )
            break

        left = coalesced[best_idx]
        right = coalesced[best_idx + 1]
        coalesced[best_idx : best_idx + 2] = [
            Segment(
                start=left.start,
                end=right.end,
                label="music",
                score=max(left.score, right.score),
            )
        ]

    if len(coalesced) < original_count:
        logger.info(
            "Expected-song merge reduced candidates from %s to %s (target=%s)",
            original_count,
            len(coalesced),
            expected_song_count,
        )

    return coalesced


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

    # Smooth local volatility so brief dips do not fragment one song into many clips.
    smoothed_energies: List[float] = []
    smoothing_radius = 2
    for idx in range(len(energies)):
        lo = max(0, idx - smoothing_radius)
        hi = min(len(energies), idx + smoothing_radius + 1)
        smoothed_energies.append(sum(energies[lo:hi]) / max(1, hi - lo))

    threshold_on = max(250.0, statistics.median(smoothed_energies) * 1.35)
    threshold_off = threshold_on * 0.80
    segments: List[Segment] = []

    active_start: float | None = None
    above_count = 0
    below_count = 0
    start_required = 2
    stop_required = 3

    for window_idx, energy in enumerate(smoothed_energies):
        if active_start is None:
            if energy >= threshold_on:
                above_count += 1
            else:
                above_count = 0

            if above_count >= start_required:
                start_idx = max(0, window_idx - start_required + 1)
                active_start = float(start_idx)
                above_count = 0
        else:
            if energy < threshold_off:
                below_count += 1
            else:
                below_count = 0

            if below_count >= stop_required:
                end_idx = max(0, window_idx - stop_required + 1)
                segments.append(Segment(active_start, float(end_idx), label="music", score=0.35))
                active_start = None
                below_count = 0

    if active_start is not None:
        segments.append(Segment(active_start, float(len(smoothed_energies)), label="music", score=0.35))

    return segments


def _get_audio_duration(audio_path: Path, logger: logging.Logger) -> float | None:
    try:
        with wave.open(str(audio_path), "rb") as wav_handle:
            sample_rate = wav_handle.getframerate()
            frame_count = wav_handle.getnframes()
        if sample_rate <= 0:
            logger.warning("Unable to read audio duration from %s (invalid sample rate)", audio_path)
            return None
        return frame_count / float(sample_rate)
    except Exception as exc:  # pragma: no cover - depends on input file
        logger.warning("Unable to read audio duration for end exclusion: %s", exc)
        return None


def _apply_exclude_window(
    segments: List[Segment],
    exclude_start_seconds: float,
    exclude_end_seconds: float,
    audio_duration_sec: float | None,
    logger: logging.Logger,
) -> List[Segment]:
    filtered = segments

    if exclude_start_seconds and exclude_start_seconds > 0.0:
        trimmed: List[Segment] = []
        for seg in filtered:
            if seg.end <= exclude_start_seconds:
                # Entire segment falls within excluded head - drop it
                continue
            # Clamp start to the exclusion boundary but keep end in original timeline
            new_start = max(seg.start, exclude_start_seconds)
            if seg.end > new_start:
                trimmed.append(Segment(new_start, seg.end, seg.label, seg.score))
        logger.info(
            "Excluding first %.2fs from segmentation: reduced raw regions %s -> %s",
            exclude_start_seconds,
            len(filtered),
            len(trimmed),
        )
        filtered = trimmed

    if exclude_end_seconds and exclude_end_seconds > 0.0:
        if audio_duration_sec is None or audio_duration_sec <= 0.0:
            logger.warning("Exclude end requested but audio duration unavailable; skipping tail exclusion.")
        else:
            end_cutoff = max(0.0, audio_duration_sec - exclude_end_seconds)
            if end_cutoff <= 0.0:
                logger.warning(
                    "Exclude end %.2fs covers entire audio (duration %.2fs); dropping all segments.",
                    exclude_end_seconds,
                    audio_duration_sec,
                )
                return []
            trimmed: List[Segment] = []
            for seg in filtered:
                if seg.start >= end_cutoff:
                    continue
                new_end = min(seg.end, end_cutoff)
                if new_end > seg.start:
                    trimmed.append(Segment(seg.start, new_end, seg.label, seg.score))
            logger.info(
                "Excluding last %.2fs from segmentation: reduced raw regions %s -> %s",
                exclude_end_seconds,
                len(filtered),
                len(trimmed),
            )
            filtered = trimmed

    return filtered


def detect_music_segments(
    audio_path: Path,
    min_segment_sec: float,
    max_segment_sec: float,
    merge_gap_sec: float,
    expected_song_count: int | None,
    logger: logging.Logger,
    merge_max_segment_sec: float | None = None,
    exclude_start_seconds: float = 0.0,
    exclude_end_seconds: float = 0.0,
) -> List[Segment]:
    """Detect candidate music regions from the working WAV file."""
    base_segments: List[Segment]

    try:
        base_segments = _segments_with_ina(audio_path)
        logger.info("inaSpeechSegmenter produced %s raw music regions", len(base_segments))
    except Exception as exc:  # pragma: no cover - depends on optional package
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        hint = ""
        if isinstance(exc, ModuleNotFoundError):
            hint = (
                " Install optional ML deps in a Python 3.9-3.12 environment "
                "(Python %s currently active)." % py_version
            )
        logger.warning("inaSpeechSegmenter unavailable or failed: %s.%s", exc, hint)
        base_segments = _segments_with_energy_fallback(audio_path, logger)

    audio_duration_sec = None
    if exclude_end_seconds and exclude_end_seconds > 0.0:
        audio_duration_sec = _get_audio_duration(audio_path, logger)

    base_segments = _apply_exclude_window(
        base_segments,
        exclude_start_seconds=exclude_start_seconds,
        exclude_end_seconds=exclude_end_seconds,
        audio_duration_sec=audio_duration_sec,
        logger=logger,
    )

    merged = merge_adjacent_segments(
        base_segments,
        max_gap_sec=merge_gap_sec,
        min_segment_sec=min_segment_sec,
        max_segment_sec=max_segment_sec,
        merge_max_segment_sec=merge_max_segment_sec,
        logger=logger,
    )

    if expected_song_count is not None:
        merged = coalesce_segments_to_expected_count(
            merged,
            expected_song_count=expected_song_count,
            merge_gap_sec=merge_gap_sec,
            max_segment_sec=max_segment_sec,
            merge_max_segment_sec=merge_max_segment_sec,
            logger=logger,
        )

    logger.info("Segmentation finalized %s candidate regions", len(merged))
    return merged
