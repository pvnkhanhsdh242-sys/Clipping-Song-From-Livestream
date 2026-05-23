"""CLI entrypoint and end-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

from app.align.whisperx_align import WhisperXRefiner
from app.clip.cutter import export_clip
from app.config import AppConfig, load_config
from app.identify.acoustid_client import AcoustIDClient
from app.identify.chromaprint_match import ChromaprintMatcher
from app.identify.types import MatchResult
from app.ingest.youtube import (
    SourceVideo,
    download_youtube_video,
    probe_youtube_metadata,
    register_local_video,
)
from app.output.manifest import ManifestRecord, write_manifests
from app.output.preview import PreviewRecord, generate_snapshots
from app.preprocess.extract_audio import extract_working_audio
from app.segment.music_segments import calculate_music_ratio, detect_music_segments
from app.singing.scorer import SingingCandidateScorer
from app.utils.ffmpeg import ensure_ffmpeg_available
from app.utils.logging import setup_logger
from app.utils.paths import prepare_output_dirs
from app.utils.timecode import sanitize_filename_component


@dataclass
class SegmentAnalysis:
    index: int
    raw_start: float
    raw_end: float
    padded_start: float
    padded_end: float
    refined_start: float
    refined_end: float
    match: MatchResult | None
    confidence: float
    boundary_method: str
    refinement_method: str
    music_ratio: float
    fingerprint_confidence: float
    duration_score: float
    boundary_quality_score: float
    final_score: float
    merge_count: int
    bridged_gap_total_sec: float
    needs_review: bool
    review_reason: str | None
    singing_score: float | None = None
    singing_model: str = "none"
    singing_decision: str = "not_scored"

    def to_preview(self) -> PreviewRecord:
        return PreviewRecord(
            index=self.index,
            start_sec=self.refined_start,
            end_sec=self.refined_end,
            song=self.match.song if self.match else "Unknown",
            artist=self.match.artist if self.match else "Unknown",
            confidence=self.final_score,
            backend=self.match.backend if self.match else "none",
            final_score=self.final_score,
            needs_review=self.needs_review,
            review_reason=self.review_reason,
            boundary_method=self.boundary_method,
            refinement_method=self.refinement_method,
            music_ratio=self.music_ratio,
            singing_score=self.singing_score,
            singing_model=self.singing_model,
            singing_decision=self.singing_decision,
        )


@dataclass
class PreviewResult:
    records: list[PreviewRecord]
    source_video: Path
    output_root: Path
    preview_dir: Path
    snapshots: list[Path]


def _split_oversized_analyses(
    analyses: list[SegmentAnalysis],
    max_segment_sec: float,
    allow_hard_split: bool,
    logger,
) -> list[SegmentAnalysis]:
    if max_segment_sec <= 0:
        return analyses

    if not allow_hard_split:
        for analysis in analyses:
            duration = analysis.refined_end - analysis.refined_start
            if duration > max_segment_sec:
                analysis.needs_review = True
                if not analysis.review_reason:
                    analysis.review_reason = "segment_exceeds_max_duration"
        return analyses

    split: list[SegmentAnalysis] = []
    next_index = 1

    for analysis in analyses:
        base_start = analysis.refined_start if analysis.refined_end > analysis.refined_start else analysis.padded_start
        base_end = analysis.refined_end if analysis.refined_end > analysis.refined_start else analysis.padded_end

        if base_end <= base_start:
            base_start, base_end = analysis.padded_start, analysis.padded_end

        cursor = base_start
        chunk_count = 0
        while cursor < base_end:
            next_end = min(cursor + max_segment_sec, base_end)
            split.append(
                SegmentAnalysis(
                    index=next_index,
                    raw_start=cursor,
                    raw_end=next_end,
                    padded_start=cursor,
                    padded_end=next_end,
                    refined_start=cursor,
                    refined_end=next_end,
                    match=analysis.match,
                    confidence=analysis.confidence,
                    boundary_method=analysis.boundary_method,
                    refinement_method=analysis.refinement_method,
                    music_ratio=analysis.music_ratio,
                    fingerprint_confidence=analysis.fingerprint_confidence,
                    duration_score=analysis.duration_score,
                    boundary_quality_score=analysis.boundary_quality_score,
                    final_score=analysis.final_score,
                    merge_count=analysis.merge_count,
                    bridged_gap_total_sec=analysis.bridged_gap_total_sec,
                    needs_review=analysis.needs_review,
                    review_reason=analysis.review_reason,
                    singing_score=analysis.singing_score,
                    singing_model=analysis.singing_model,
                    singing_decision=analysis.singing_decision,
                )
            )
            next_index += 1
            chunk_count += 1
            cursor = next_end

        if chunk_count > 1:
            logger.info(
                "Split oversized segment %.3f -> %.3f into %s clips capped at %.2fs",
                base_start,
                base_end,
                chunk_count,
                max_segment_sec,
            )

    return split


def _filter_analyses_by_music_ratio(
    analyses: list[SegmentAnalysis],
    music_ratio_threshold: float,
    logger,
) -> list[SegmentAnalysis]:
    if music_ratio_threshold <= 0:
        return analyses

    kept: list[SegmentAnalysis] = []
    for analysis in analyses:
        if analysis.music_ratio >= music_ratio_threshold:
            kept.append(analysis)
            continue

        logger.info(
            "Filtered segment %s by music ratio %.4f < %.4f (%.3f -> %.3f)",
            analysis.index,
            analysis.music_ratio,
            music_ratio_threshold,
            analysis.refined_start,
            analysis.refined_end,
        )

    if len(kept) != len(analyses):
        logger.info(
            "Music-ratio filter kept %s/%s segments (threshold=%.3f)",
            len(kept),
            len(analyses),
            music_ratio_threshold,
        )

    for idx, analysis in enumerate(kept, start=1):
        analysis.index = idx

    return kept


def _filter_analyses_by_singing_score(
    analyses: list[SegmentAnalysis],
    singing_model_mode: str,
    singing_score_threshold: float,
    logger,
) -> list[SegmentAnalysis]:
    if singing_model_mode != "filter":
        return analyses

    kept: list[SegmentAnalysis] = []
    for analysis in analyses:
        if analysis.singing_score is None or analysis.singing_score >= singing_score_threshold:
            kept.append(analysis)
            continue

        logger.info(
            "Filtered segment %s by singing score %.4f < %.4f (%.3f -> %.3f)",
            analysis.index,
            analysis.singing_score,
            singing_score_threshold,
            analysis.refined_start,
            analysis.refined_end,
        )

    if len(kept) != len(analyses):
        logger.info(
            "Singing-score filter kept %s/%s segments (threshold=%.3f)",
            len(kept),
            len(analyses),
            singing_score_threshold,
        )

    for idx, analysis in enumerate(kept, start=1):
        analysis.index = idx

    return kept


def _resolve_source(config: AppConfig, vods_dir: Path, logger) -> SourceVideo:
    if config.url:
        return download_youtube_video(config.url, vods_dir, logger)
    assert config.file is not None
    return register_local_video(config.file)


def _build_run_label(title: str, video_id: str) -> str:
    raw_title = title.strip()
    raw_id = video_id.strip()
    raw_label = raw_title or raw_id or "unknown"
    return sanitize_filename_component(raw_label)


def _resolve_run_output_root(config: AppConfig) -> Path:
    if config.url:
        video_id, title = probe_youtube_metadata(config.url)
        label = _build_run_label(title, video_id)
    else:
        assert config.file is not None
        stem = config.file.stem
        label = _build_run_label(stem, stem)
    return config.outdir / label


def _analyze_segments(
    segments,
    raw_segments,
    audio_duration_sec: float | None,
    working_audio: Path,
    output_dirs: dict[str, Path],
    matcher: ChromaprintMatcher,
    acoustid: AcoustIDClient,
    refiner: WhisperXRefiner,
    singing_scorer: SingingCandidateScorer,
    config: AppConfig,
    logger,
) -> list[SegmentAnalysis]:
    analyses: list[SegmentAnalysis] = []

    def _duration_score(duration: float) -> float:
        if 60.0 <= duration <= 360.0:
            return 1.0
        if 30.0 <= duration < 60.0 or 360.0 < duration <= 600.0:
            return 0.7
        return 0.4

    for idx, segment in enumerate(segments, start=1):
        logger.info("Processing segment %s: %.3f -> %.3f", idx, segment.start, segment.end)

        raw_start = segment.raw_start if segment.raw_start is not None else segment.start
        raw_end = segment.raw_end if segment.raw_end is not None else segment.end
        padded_start = segment.start
        padded_end = segment.end

        match = matcher.match_segment(working_audio, segment.start, segment.end, output_dirs["tmp"])
        match_warning = None
        if match is not None and match.backend == "chromaprint_unavailable":
            match_warning = match.review_reason
            match = None

        if match is None:
            match = acoustid.identify_segment(working_audio, segment.start, segment.end, output_dirs["tmp"])

        refined_start, refined_end = refiner.refine_segment(
            working_audio,
            segment.start,
            segment.end,
            output_dirs["tmp"],
            mode=config.whisperx_boundary_mode,
            max_start_shrink_sec=config.whisperx_max_start_shrink_sec,
            max_end_shrink_sec=config.whisperx_max_end_shrink_sec,
            post_roll_sec=0.0,
            audio_duration_sec=audio_duration_sec,
        )

        if refined_end <= refined_start:
            refined_start, refined_end = segment.start, segment.end

        refinement_method = "none"
        if config.whisperx_boundary_mode == "metadata":
            refinement_method = "whisperx_metadata"
        elif config.whisperx_boundary_mode == "safe":
            refinement_method = "whisperx_safe"

        fingerprint_confidence = match.confidence if match else 0.0
        confidence = fingerprint_confidence if match else segment.score
        music_ratio = calculate_music_ratio(raw_segments, refined_start, refined_end)
        duration_score = _duration_score(refined_end - refined_start)
        boundary_quality_score = 1.0
        if segment.boundary_method.startswith("energy_fallback"):
            boundary_quality_score -= 0.2
        if segment.merge_count > 0:
            boundary_quality_score -= 0.05
        if segment.needs_review:
            boundary_quality_score -= 0.3
        boundary_quality_score = max(0.0, min(1.0, boundary_quality_score))

        singing_result = singing_scorer.score_candidate(
            working_audio,
            refined_start,
            refined_end,
            music_ratio=music_ratio,
            fingerprint_confidence=fingerprint_confidence,
            duration_score=duration_score,
            boundary_quality_score=boundary_quality_score,
            merge_count=segment.merge_count,
            bridged_gap_total_sec=segment.bridged_gap_total_sec,
            boundary_method=segment.boundary_method,
        )
        singing_score = singing_result.score

        if singing_score is None:
            final_score = (
                0.40 * music_ratio
                + 0.35 * fingerprint_confidence
                + 0.15 * duration_score
                + 0.10 * boundary_quality_score
            )
        else:
            final_score = (
                0.30 * music_ratio
                + 0.25 * fingerprint_confidence
                + 0.20 * singing_score
                + 0.15 * duration_score
                + 0.10 * boundary_quality_score
            )

        needs_review = bool(segment.needs_review)
        review_reason = segment.review_reason
        if match and match.needs_review:
            needs_review = True
            review_reason = review_reason or match.review_reason
        if match_warning:
            needs_review = True
            review_reason = review_reason or match_warning
        if refined_end - refined_start > config.max_segment and not config.allow_hard_split:
            needs_review = True
            review_reason = review_reason or "segment_exceeds_max_duration"
        if singing_score is not None and singing_score < config.singing_score_threshold:
            needs_review = True
            review_reason = review_reason or "low_singing_score"
        if final_score < config.review_score_threshold:
            needs_review = True
            review_reason = review_reason or "low_score"

        analyses.append(
            SegmentAnalysis(
                index=idx,
                raw_start=raw_start,
                raw_end=raw_end,
                padded_start=padded_start,
                padded_end=padded_end,
                refined_start=refined_start,
                refined_end=refined_end,
                match=match,
                confidence=float(confidence),
                boundary_method=segment.boundary_method,
                refinement_method=refinement_method,
                music_ratio=music_ratio,
                fingerprint_confidence=float(fingerprint_confidence),
                duration_score=duration_score,
                boundary_quality_score=boundary_quality_score,
                final_score=float(final_score),
                merge_count=int(segment.merge_count),
                bridged_gap_total_sec=float(segment.bridged_gap_total_sec),
                needs_review=needs_review,
                review_reason=review_reason,
                singing_score=float(singing_score) if singing_score is not None else None,
                singing_model=singing_result.model_name,
                singing_decision=singing_result.decision,
            )
        )

    return analyses


def _maybe_upload_outputs(
    config: AppConfig,
    run_output_root: Path,
    logger,
    records: Sequence[ManifestRecord] | None = None,
) -> None:
    if not config.gdrive_upload:
        return

    if not config.gdrive_folder_id:
        logger.warning("Google Drive upload requested but gdrive_folder_id is missing; skipping upload.")
        return

    try:
        if getattr(config, "gdrive_upload_mode", "clips") == "all":
            from app.integrations.gdrive import upload_output_dir

            upload_output_dir(
                output_dir=run_output_root,
                parent_folder_id=config.gdrive_folder_id,
                client_secrets_path=config.gdrive_client_secrets,
                token_path=config.gdrive_token_path,
                include_tmp=config.gdrive_include_tmp,
                logger=logger,
            )
        else:
            # Upload only the clips folder to Drive; other artifacts remain local.
            from app.integrations.gdrive import upload_clips_dir

            clip_files = None
            if records:
                clip_files = [Path(record.clip_path) for record in records if record.clip_path]

            upload_clips_dir(
                output_dir=run_output_root,
                parent_folder_id=config.gdrive_folder_id,
                client_secrets_path=config.gdrive_client_secrets,
                token_path=config.gdrive_token_path,
                logger=logger,
                clip_files=clip_files,
            )
    except Exception as exc:  # pragma: no cover - network/auth dependent path
        logger.warning("Google Drive upload failed: %s", exc)


def run_pipeline(
    config: AppConfig,
    progress_callback: Callable[[int, int, float, float], None] | None = None,
) -> int:
    run_output_root = _resolve_run_output_root(config)
    output_dirs = prepare_output_dirs(run_output_root)
    log_path = output_dirs["logs"] / f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_path)

    logger.info("Starting karaoke-clipper pipeline")
    ensure_ffmpeg_available()

    source = _resolve_source(config, output_dirs["vods"], logger)
    logger.info("Source mode=%s, video=%s", source.source_mode, source.video_path)

    working_audio = output_dirs["audio"] / f"{source.video_id}.wav"
    extract_working_audio(source.video_path, working_audio, config.sample_rate, logger)

    segmentation = detect_music_segments(
        audio_path=working_audio,
        min_segment_sec=config.min_segment,
        max_segment_sec=config.max_segment,
        merge_gap_sec=config.merge_gap,
        expected_song_count=config.expected_song_count,
        logger=logger,
        merge_max_segment_sec=config.merge_max_segment,
        segment_tolerance_sec=config.segment_tolerance,
        pre_roll_sec=config.pre_roll_sec,
        post_roll_sec=config.post_roll_sec,
        bridge_noise_gap_sec=config.bridge_noise_gap_sec,
        bridge_speech_gap_sec=config.bridge_speech_gap_sec,
        allow_hard_split=config.allow_hard_split,
        energy_frame_ms=config.energy_frame_ms,
        energy_min_active_ms=config.energy_min_active_ms,
        energy_min_silence_ms=config.energy_min_silence_ms,
        exclude_start_seconds=config.exclude_start_seconds,
        exclude_end_seconds=config.exclude_end_seconds,
    )

    matcher = ChromaprintMatcher(config.ref_library, config.fingerprint_threshold, logger)
    acoustid = AcoustIDClient(config.acoustid_api_key, config.use_acoustid, logger)
    refiner = WhisperXRefiner(device=config.device, logger=logger)
    singing_scorer = SingingCandidateScorer.from_config(config, logger)

    try:
        analyses = _analyze_segments(
            segmentation.segments,
            segmentation.raw_segments,
            segmentation.audio_duration_sec,
            working_audio,
            output_dirs,
            matcher,
            acoustid,
            refiner,
            singing_scorer,
            config,
            logger,
        )
    finally:
        refiner.release()

    analyses = _split_oversized_analyses(analyses, config.max_segment, config.allow_hard_split, logger)
    analyses = _filter_analyses_by_music_ratio(analyses, config.music_ratio_threshold, logger)
    analyses = _filter_analyses_by_singing_score(
        analyses,
        config.singing_model_mode,
        config.singing_score_threshold,
        logger,
    )
    records: list[ManifestRecord] = []

    total_clips = len(analyses)
    for analysis in analyses:
        label = f"song_{analysis.index:03d}"
        if analysis.match is not None:
            label = sanitize_filename_component(
                f"{analysis.match.artist} - {analysis.match.song}__{analysis.index:03d}"
            )

        logger.info(
            "Exporting clip %s/%s: %.3f -> %.3f",
            analysis.index,
            total_clips,
            analysis.refined_start,
            analysis.refined_end,
        )

        if progress_callback is not None:
            progress_callback(
                analysis.index,
                total_clips,
                analysis.refined_start,
                analysis.refined_end,
            )

        clip = export_clip(
            video_path=source.video_path,
            start_sec=analysis.refined_start,
            end_sec=analysis.refined_end,
            clips_dir=output_dirs["clips"],
            clip_stem=label,
            include_audio_clip=config.audio_clips,
            mode=config.clip_mode,
            clip_resolution=config.clip_resolution,
            logger=logger,
        )

        records.append(
            ManifestRecord(
                source_video=str(source.video_path),
                video_id=source.video_id,
                song=analysis.match.song if analysis.match else "Unknown",
                artist=analysis.match.artist if analysis.match else "Unknown",
                raw_start_sec=round(analysis.raw_start, 3),
                raw_end_sec=round(analysis.raw_end, 3),
                start_sec=round(analysis.refined_start, 3),
                end_sec=round(analysis.refined_end, 3),
                duration_sec=round(analysis.refined_end - analysis.refined_start, 3),
                pre_roll_sec=round(config.pre_roll_sec, 3),
                post_roll_sec=round(config.post_roll_sec, 3),
                boundary_method=analysis.boundary_method,
                refinement_method=analysis.refinement_method,
                music_ratio=round(analysis.music_ratio, 4),
                fingerprint_confidence=round(analysis.fingerprint_confidence, 4),
                duration_score=round(analysis.duration_score, 4),
                boundary_quality_score=round(analysis.boundary_quality_score, 4),
                final_score=round(analysis.final_score, 4),
                merge_count=analysis.merge_count,
                bridged_gap_total_sec=round(analysis.bridged_gap_total_sec, 3),
                needs_review=analysis.needs_review,
                review_reason=analysis.review_reason,
                confidence=round(analysis.confidence, 4),
                clip_path=str(clip.clip_path),
                audio_path=str(clip.audio_path) if clip.audio_path else None,
                backend=analysis.match.backend if analysis.match else "none",
                singing_score=round(analysis.singing_score, 4) if analysis.singing_score is not None else None,
                singing_model=analysis.singing_model,
                singing_decision=analysis.singing_decision,
            )
        )

    manifest_base = output_dirs["manifests"] / f"{source.video_id}_manifest"
    json_path, csv_path = write_manifests(records, manifest_base)

    logger.info("Pipeline finished with %s clips", len(records))
    logger.info("Manifest JSON: %s", json_path)
    logger.info("Manifest CSV: %s", csv_path)
    _maybe_upload_outputs(config, run_output_root, logger, records)
    return 0


def preview_pipeline(config: AppConfig, snapshot_limit: int = 0) -> PreviewResult:
    run_output_root = _resolve_run_output_root(config)
    output_dirs = prepare_output_dirs(run_output_root)
    log_path = output_dirs["logs"] / f"preview_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_path, name="karaoke_clipper_preview")

    logger.info("Starting karaoke-clipper preview")
    ensure_ffmpeg_available()

    source = _resolve_source(config, output_dirs["vods"], logger)
    logger.info("Source mode=%s, video=%s", source.source_mode, source.video_path)

    working_audio = output_dirs["audio"] / f"{source.video_id}.wav"
    extract_working_audio(source.video_path, working_audio, config.sample_rate, logger)

    segmentation = detect_music_segments(
        audio_path=working_audio,
        min_segment_sec=config.min_segment,
        max_segment_sec=config.max_segment,
        merge_gap_sec=config.merge_gap,
        expected_song_count=config.expected_song_count,
        logger=logger,
        merge_max_segment_sec=config.merge_max_segment,
        segment_tolerance_sec=config.segment_tolerance,
        pre_roll_sec=config.pre_roll_sec,
        post_roll_sec=config.post_roll_sec,
        bridge_noise_gap_sec=config.bridge_noise_gap_sec,
        bridge_speech_gap_sec=config.bridge_speech_gap_sec,
        allow_hard_split=config.allow_hard_split,
        energy_frame_ms=config.energy_frame_ms,
        energy_min_active_ms=config.energy_min_active_ms,
        energy_min_silence_ms=config.energy_min_silence_ms,
        exclude_start_seconds=config.exclude_start_seconds,
        exclude_end_seconds=config.exclude_end_seconds,
    )

    matcher = ChromaprintMatcher(config.ref_library, config.fingerprint_threshold, logger)
    acoustid = AcoustIDClient(config.acoustid_api_key, config.use_acoustid, logger)
    refiner = WhisperXRefiner(device=config.device, logger=logger)
    singing_scorer = SingingCandidateScorer.from_config(config, logger)

    try:
        analyses = _analyze_segments(
            segmentation.segments,
            segmentation.raw_segments,
            segmentation.audio_duration_sec,
            working_audio,
            output_dirs,
            matcher,
            acoustid,
            refiner,
            singing_scorer,
            config,
            logger,
        )
    finally:
        refiner.release()

    analyses = _split_oversized_analyses(analyses, config.max_segment, config.allow_hard_split, logger)
    analyses = _filter_analyses_by_music_ratio(analyses, config.music_ratio_threshold, logger)
    analyses = _filter_analyses_by_singing_score(
        analyses,
        config.singing_model_mode,
        config.singing_score_threshold,
        logger,
    )
    preview_records = [analysis.to_preview() for analysis in analyses]
    logger.info("Preview generated %s candidate segments", len(preview_records))

    snapshots: list[Path] = []
    if snapshot_limit > 0:
        snapshots = generate_snapshots(
            video_path=source.video_path,
            records=preview_records,
            output_dir=output_dirs["previews"],
            logger=logger,
            limit=snapshot_limit,
        )

    return PreviewResult(
        records=preview_records,
        source_video=source.video_path,
        output_root=run_output_root,
        preview_dir=output_dirs["previews"],
        snapshots=snapshots,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config(argv)
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
