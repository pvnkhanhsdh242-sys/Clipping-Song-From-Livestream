"""Music candidate segmentation stage."""

from __future__ import annotations

import logging
import math
import statistics
import sys
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RawAudioSegment:
    label: str
    start: float
    end: float


@dataclass
class Segment:
    start: float
    end: float
    label: str = "music"
    score: float = 0.0
    raw_start: float | None = None
    raw_end: float | None = None
    merge_count: int = 0
    bridged_gap_total_sec: float = 0.0
    boundary_method: str = "ina"
    needs_review: bool = False
    review_reason: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class SegmentationResult:
    segments: List[Segment]
    raw_segments: List[RawAudioSegment]
    audio_duration_sec: float | None


def _clone_segment(segment: Segment, start: float | None = None, end: float | None = None) -> Segment:
    return Segment(
        start=segment.start if start is None else start,
        end=segment.end if end is None else end,
        label=segment.label,
        score=segment.score,
        raw_start=segment.raw_start,
        raw_end=segment.raw_end,
        merge_count=segment.merge_count,
        bridged_gap_total_sec=segment.bridged_gap_total_sec,
        boundary_method=segment.boundary_method,
        needs_review=segment.needs_review,
        review_reason=segment.review_reason,
    )


def calculate_music_ratio(raw_segments: List[RawAudioSegment], start_sec: float, end_sec: float) -> float:
    if end_sec <= start_sec:
        return 0.0

    music_total = 0.0
    for seg in raw_segments:
        if "music" not in seg.label.lower():
            continue
        overlap_start = max(start_sec, seg.start)
        overlap_end = min(end_sec, seg.end)
        if overlap_end > overlap_start:
            music_total += overlap_end - overlap_start

    duration = end_sec - start_sec
    if duration <= 0:
        return 0.0
    return min(1.0, max(0.0, music_total / duration))


def merge_adjacent_segments(
    segments: List[Segment],
    max_gap_sec: float,
    min_segment_sec: float,
    max_segment_sec: float,
    merge_max_segment_sec: float | None = None,
    segment_tolerance_sec: float = 0.0,
    allow_hard_split: bool = False,
    logger: logging.Logger | None = None,
) -> List[Segment]:
    """Merge nearby segments and enforce min/max duration limits."""
    if not segments:
        return []

    tolerance_sec = max(0.0, float(segment_tolerance_sec))
    effective_max_segment = max_segment_sec
    effective_merge_max = merge_max_segment_sec if merge_max_segment_sec is not None else max_segment_sec
    if merge_max_segment_sec is not None and merge_max_segment_sec != max_segment_sec:
        effective_max_segment = merge_max_segment_sec
        effective_merge_max = merge_max_segment_sec
        if logger is not None:
            logger.warning(
                "merge_max_segment (%.2fs) overrides max_segment (%.2fs) for merging and splitting.",
                merge_max_segment_sec,
                max_segment_sec,
            )

    merge_cap_sec = effective_merge_max + tolerance_sec
    max_effective_sec = effective_max_segment + tolerance_sec
    min_effective_sec = max(0.0, min_segment_sec - tolerance_sec)

    ordered = sorted(segments, key=lambda s: s.start)
    merged: List[Segment] = [_clone_segment(ordered[0])]

    for seg in ordered[1:]:
        current = merged[-1]
        gap = seg.start - current.end
        combined_duration = max(current.end, seg.end) - current.start
        if gap <= max_gap_sec and combined_duration <= merge_cap_sec:
            current.end = max(current.end, seg.end)
            current.raw_start = current.raw_start if current.raw_start is not None else current.start
            current.raw_end = max(current.raw_end or current.end, seg.raw_end or seg.end)
            current.score = max(current.score, seg.score)
            current.merge_count += seg.merge_count
            current.bridged_gap_total_sec += seg.bridged_gap_total_sec
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
            merged.append(_clone_segment(seg))

    clipped: List[Segment] = []
    for seg in merged:
        if seg.duration < min_effective_sec:
            continue

        if seg.duration <= max_effective_sec:
            clipped.append(seg)
            continue

        if not allow_hard_split:
            seg.needs_review = True
            seg.review_reason = "segment_exceeds_max_duration"
            clipped.append(seg)
            continue

        chunks: List[Segment] = []
        cursor = seg.start
        while cursor < seg.end:
            next_end = min(cursor + effective_max_segment, seg.end)
            chunk = _clone_segment(seg, start=cursor, end=next_end)
            chunk.raw_start = cursor
            chunk.raw_end = next_end
            chunks.append(chunk)
            cursor = next_end

        if len(chunks) > 1:
            tail = chunks[-1]
            if tail.duration < min_effective_sec:
                prior = chunks[-2]
                combined_duration = tail.end - prior.start
                if combined_duration <= max_effective_sec:
                    prior.end = tail.end
                    prior.raw_end = tail.end
                    prior.score = max(prior.score, tail.score)
                    chunks.pop()
                else:
                    chunks.pop()

        for chunk in chunks:
            if chunk.duration >= min_effective_sec:
                clipped.append(chunk)

    return clipped


def coalesce_segments_to_expected_count(
    segments: List[Segment],
    expected_song_count: int,
    merge_gap_sec: float,
    max_segment_sec: float,
    logger: logging.Logger,
    merge_max_segment_sec: float | None = None,
    segment_tolerance_sec: float = 0.0,
) -> List[Segment]:
    """Best-effort merge of nearest neighbors until target song count is reached."""
    if expected_song_count <= 0 or len(segments) <= expected_song_count:
        return segments

    tolerance_sec = max(0.0, float(segment_tolerance_sec))
    merge_cap_sec = (
        merge_max_segment_sec if merge_max_segment_sec is not None else max_segment_sec
    ) + tolerance_sec

    coalesced = [_clone_segment(s) for s in sorted(segments, key=lambda seg: seg.start)]
    original_count = len(coalesced)

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
        merged_seg = _clone_segment(left, end=right.end)
        merged_seg.raw_start = left.raw_start if left.raw_start is not None else left.start
        merged_seg.raw_end = max(left.raw_end or left.end, right.raw_end or right.end)
        merged_seg.score = max(left.score, right.score)
        merged_seg.merge_count += right.merge_count
        merged_seg.bridged_gap_total_sec += right.bridged_gap_total_sec
        coalesced[best_idx : best_idx + 2] = [merged_seg]

    if len(coalesced) < original_count:
        logger.info(
            "Expected-song merge reduced candidates from %s to %s (target=%s)",
            original_count,
            len(coalesced),
            expected_song_count,
        )

    return coalesced


def _raw_segments_with_ina(audio_path: Path) -> List[RawAudioSegment]:
    from inaSpeechSegmenter import Segmenter  # type: ignore

    segmenter = Segmenter(detect_gender=False)
    raw = segmenter(str(audio_path))

    segments: List[RawAudioSegment] = []
    for label, start, end in raw:
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        segments.append(RawAudioSegment(label=str(label), start=start_f, end=end_f))

    return segments


def _raw_segments_with_energy_fallback(
    audio_path: Path,
    logger: logging.Logger,
    energy_frame_ms: int,
    energy_min_active_ms: int,
    energy_min_silence_ms: int,
) -> List[RawAudioSegment]:
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

    frame_size = max(int(sample_rate * (energy_frame_ms / 1000.0)), 1)
    frame_duration = frame_size / float(sample_rate)
    energies: List[float] = []

    for idx in range(0, len(samples), frame_size):
        chunk = samples[idx : idx + frame_size]
        if not chunk:
            continue
        energies.append(sum(abs(value) for value in chunk) / len(chunk))

    if not energies:
        return []

    smoothed_energies: List[float] = []
    smoothing_radius = 2
    for idx in range(len(energies)):
        lo = max(0, idx - smoothing_radius)
        hi = min(len(energies), idx + smoothing_radius + 1)
        smoothed_energies.append(sum(energies[lo:hi]) / max(1, hi - lo))

    threshold_on = max(250.0, statistics.median(smoothed_energies) * 1.35)
    threshold_off = threshold_on * 0.80
    segments: List[RawAudioSegment] = []

    active_start: float | None = None
    above_count = 0
    below_count = 0
    start_required = max(1, int(math.ceil(energy_min_active_ms / float(energy_frame_ms))))
    stop_required = max(1, int(math.ceil(energy_min_silence_ms / float(energy_frame_ms))))

    for window_idx, energy in enumerate(smoothed_energies):
        timestamp = window_idx * frame_duration
        if active_start is None:
            if energy >= threshold_on:
                above_count += 1
            else:
                above_count = 0

            if above_count >= start_required:
                start_idx = max(0, window_idx - start_required + 1)
                active_start = float(start_idx) * frame_duration
                above_count = 0
        else:
            if energy < threshold_off:
                below_count += 1
            else:
                below_count = 0

            if below_count >= stop_required:
                end_idx = max(0, window_idx - stop_required + 1)
                end_sec = float(end_idx) * frame_duration
                if end_sec > active_start:
                    segments.append(RawAudioSegment(label="music", start=active_start, end=end_sec))
                active_start = None
                below_count = 0

    if active_start is not None:
        end_sec = float(len(smoothed_energies)) * frame_duration
        segments.append(RawAudioSegment(label="music", start=active_start, end=end_sec))

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
        logger.warning("Unable to read audio duration for padding/exclusion: %s", exc)
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
                continue
            new_start = max(seg.start, exclude_start_seconds)
            if seg.end > new_start:
                adjusted = _clone_segment(seg, start=new_start)
                adjusted.raw_start = new_start
                trimmed.append(adjusted)
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
            trimmed = []
            for seg in filtered:
                if seg.start >= end_cutoff:
                    continue
                new_end = min(seg.end, end_cutoff)
                if new_end > seg.start:
                    adjusted = _clone_segment(seg, end=new_end)
                    adjusted.raw_end = new_end
                    trimmed.append(adjusted)
            logger.info(
                "Excluding last %.2fs from segmentation: reduced raw regions %s -> %s",
                exclude_end_seconds,
                len(filtered),
                len(trimmed),
            )
            filtered = trimmed

    return filtered


def apply_padding(
    segments: List[Segment],
    audio_duration_sec: float | None,
    pre_roll_sec: float,
    post_roll_sec: float,
) -> List[Segment]:
    if pre_roll_sec <= 0 and post_roll_sec <= 0:
        return segments

    padded: List[Segment] = []
    for seg in segments:
        new_start = max(0.0, seg.start - pre_roll_sec)
        new_end = seg.end + post_roll_sec
        if audio_duration_sec is not None:
            new_end = min(audio_duration_sec, new_end)
        padded.append(_clone_segment(seg, start=new_start, end=new_end))

    return padded


def build_music_candidates_from_raw_timeline(
    raw_segments: List[RawAudioSegment],
    bridge_noise_gap_sec: float,
    bridge_speech_gap_sec: float,
    logger: logging.Logger,
    source_label: str,
) -> List[Segment]:
    ordered = sorted(raw_segments, key=lambda s: s.start)
    candidates: List[Segment] = []

    current_start: float | None = None
    current_end: float | None = None
    pending_gap_label: str | None = None
    merge_count = 0
    bridged_gap_total = 0.0

    for seg in ordered:
        label = seg.label.lower()
        is_music = "music" in label
        if is_music:
            if current_start is None:
                current_start = seg.start
                current_end = seg.end
                pending_gap_label = None
                continue

            gap_duration = max(0.0, seg.start - (current_end or seg.start))
            should_bridge = False
            if pending_gap_label is None:
                should_bridge = gap_duration <= 0.0
            elif pending_gap_label == "noise" and gap_duration <= bridge_noise_gap_sec:
                should_bridge = True
            elif pending_gap_label in {"speech", "male", "female"} and gap_duration <= bridge_speech_gap_sec:
                should_bridge = True

            if should_bridge:
                if gap_duration > 0:
                    merge_count += 1
                    bridged_gap_total += gap_duration
                current_end = max(current_end or seg.end, seg.end)
            else:
                boundary_method = source_label
                if merge_count or bridged_gap_total > 0:
                    boundary_method = f"{source_label}_bridge"
                candidates.append(
                    Segment(
                        start=current_start,
                        end=current_end or current_start,
                        label="music",
                        score=1.0,
                        raw_start=current_start,
                        raw_end=current_end or current_start,
                        merge_count=merge_count,
                        bridged_gap_total_sec=bridged_gap_total,
                        boundary_method=boundary_method,
                    )
                )
                current_start = seg.start
                current_end = seg.end
                merge_count = 0
                bridged_gap_total = 0.0

            pending_gap_label = None
        else:
            if current_start is not None:
                pending_gap_label = label

    if current_start is not None:
        boundary_method = source_label
        if merge_count or bridged_gap_total > 0:
            boundary_method = f"{source_label}_bridge"
        candidates.append(
            Segment(
                start=current_start,
                end=current_end or current_start,
                label="music",
                score=1.0,
                raw_start=current_start,
                raw_end=current_end or current_start,
                merge_count=merge_count,
                bridged_gap_total_sec=bridged_gap_total,
                boundary_method=boundary_method,
            )
        )

    logger.info("Built %s candidates from %s raw segments", len(candidates), len(raw_segments))
    return candidates


def detect_music_segments(
    audio_path: Path,
    min_segment_sec: float,
    max_segment_sec: float,
    merge_gap_sec: float,
    expected_song_count: int | None,
    logger: logging.Logger,
    merge_max_segment_sec: float | None = None,
    segment_tolerance_sec: float = 0.0,
    pre_roll_sec: float = 0.0,
    post_roll_sec: float = 0.0,
    bridge_noise_gap_sec: float = 2.0,
    bridge_speech_gap_sec: float = 1.0,
    allow_hard_split: bool = False,
    energy_frame_ms: int = 100,
    energy_min_active_ms: int = 500,
    energy_min_silence_ms: int = 1200,
    exclude_start_seconds: float = 0.0,
    exclude_end_seconds: float = 0.0,
) -> SegmentationResult:
    """Detect candidate music regions from the working WAV file."""
    raw_segments: List[RawAudioSegment]
    source_label = "ina"

    try:
        raw_segments = _raw_segments_with_ina(audio_path)
        logger.info("inaSpeechSegmenter produced %s raw regions", len(raw_segments))
    except Exception as exc:  # pragma: no cover - depends on optional package
        py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        hint = ""
        if isinstance(exc, ModuleNotFoundError):
            hint = (
                " Install optional ML deps in a Python 3.9-3.12 environment "
                "(Python %s currently active)." % py_version
            )
        logger.warning("inaSpeechSegmenter unavailable or failed: %s.%s", exc, hint)
        source_label = "energy_fallback"
        raw_segments = _raw_segments_with_energy_fallback(
            audio_path,
            logger,
            energy_frame_ms=energy_frame_ms,
            energy_min_active_ms=energy_min_active_ms,
            energy_min_silence_ms=energy_min_silence_ms,
        )

    audio_duration_sec = None
    if exclude_end_seconds > 0.0 or pre_roll_sec > 0.0 or post_roll_sec > 0.0:
        audio_duration_sec = _get_audio_duration(audio_path, logger)

    base_segments = build_music_candidates_from_raw_timeline(
        raw_segments,
        bridge_noise_gap_sec=bridge_noise_gap_sec,
        bridge_speech_gap_sec=bridge_speech_gap_sec,
        logger=logger,
        source_label=source_label,
    )

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
        segment_tolerance_sec=segment_tolerance_sec,
        allow_hard_split=allow_hard_split,
        logger=logger,
    )

    if expected_song_count is not None:
        merged = coalesce_segments_to_expected_count(
            merged,
            expected_song_count=expected_song_count,
            merge_gap_sec=merge_gap_sec,
            max_segment_sec=max_segment_sec,
            merge_max_segment_sec=merge_max_segment_sec,
            segment_tolerance_sec=segment_tolerance_sec,
            logger=logger,
        )

    padded = apply_padding(
        merged,
        audio_duration_sec=audio_duration_sec,
        pre_roll_sec=pre_roll_sec,
        post_roll_sec=post_roll_sec,
    )

    logger.info("Segmentation finalized %s candidate regions", len(padded))
    return SegmentationResult(
        segments=padded,
        raw_segments=raw_segments,
        audio_duration_sec=audio_duration_sec,
    )
