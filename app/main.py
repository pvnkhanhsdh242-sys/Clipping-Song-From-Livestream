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
from app.segment.music_segments import detect_music_segments
from app.utils.ffmpeg import ensure_ffmpeg_available
from app.utils.logging import setup_logger
from app.utils.paths import prepare_output_dirs
from app.utils.timecode import sanitize_filename_component


@dataclass
class SegmentAnalysis:
    index: int
    start: float
    end: float
    refined_start: float
    refined_end: float
    match: MatchResult | None
    confidence: float

    def to_preview(self) -> PreviewRecord:
        return PreviewRecord(
            index=self.index,
            start_sec=self.refined_start,
            end_sec=self.refined_end,
            song=self.match.song if self.match else "Unknown",
            artist=self.match.artist if self.match else "Unknown",
            confidence=self.confidence,
            backend=self.match.backend if self.match else "none",
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
    logger,
) -> list[SegmentAnalysis]:
    if max_segment_sec <= 0:
        return analyses

    split: list[SegmentAnalysis] = []
    next_index = 1

    for analysis in analyses:
        base_start = analysis.refined_start if analysis.refined_end > analysis.refined_start else analysis.start
        base_end = analysis.refined_end if analysis.refined_end > analysis.refined_start else analysis.end

        if base_end <= base_start:
            base_start, base_end = analysis.start, analysis.end

        cursor = base_start
        chunk_count = 0
        while cursor < base_end:
            next_end = min(cursor + max_segment_sec, base_end)
            split.append(
                SegmentAnalysis(
                    index=next_index,
                    start=analysis.start,
                    end=analysis.end,
                    refined_start=cursor,
                    refined_end=next_end,
                    match=analysis.match,
                    confidence=analysis.confidence,
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
    working_audio: Path,
    output_dirs: dict[str, Path],
    matcher: ChromaprintMatcher,
    acoustid: AcoustIDClient,
    refiner: WhisperXRefiner,
    logger,
) -> list[SegmentAnalysis]:
    analyses: list[SegmentAnalysis] = []

    for idx, segment in enumerate(segments, start=1):
        logger.info("Processing segment %s: %.3f -> %.3f", idx, segment.start, segment.end)

        match = matcher.match_segment(working_audio, segment.start, segment.end, output_dirs["tmp"])
        if match is None:
            match = acoustid.identify_segment(working_audio, segment.start, segment.end, output_dirs["tmp"])

        refined_start, refined_end = refiner.refine_segment(
            working_audio,
            segment.start,
            segment.end,
            output_dirs["tmp"],
        )

        if refined_end <= refined_start:
            refined_start, refined_end = segment.start, segment.end

        confidence = match.confidence if match else segment.score
        analyses.append(
            SegmentAnalysis(
                index=idx,
                start=segment.start,
                end=segment.end,
                refined_start=refined_start,
                refined_end=refined_end,
                match=match,
                confidence=float(confidence),
            )
        )

    return analyses


def _maybe_upload_outputs(config: AppConfig, run_output_root: Path, logger) -> None:
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

            upload_clips_dir(
                output_dir=run_output_root,
                parent_folder_id=config.gdrive_folder_id,
                client_secrets_path=config.gdrive_client_secrets,
                token_path=config.gdrive_token_path,
                logger=logger,
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

    segments = detect_music_segments(
        audio_path=working_audio,
        min_segment_sec=config.min_segment,
        max_segment_sec=config.max_segment,
        merge_gap_sec=config.merge_gap,
        expected_song_count=config.expected_song_count,
        logger=logger,
        merge_max_segment_sec=config.merge_max_segment,
        exclude_start_seconds=config.exclude_start_seconds,
        exclude_end_seconds=config.exclude_end_seconds,
    )

    matcher = ChromaprintMatcher(config.ref_library, config.fingerprint_threshold, logger)
    acoustid = AcoustIDClient(config.acoustid_api_key, config.use_acoustid, logger)
    refiner = WhisperXRefiner(device=config.device, logger=logger)

    analyses = _analyze_segments(segments, working_audio, output_dirs, matcher, acoustid, refiner, logger)
    analyses = _split_oversized_analyses(analyses, config.max_segment, logger)
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
                start_sec=round(analysis.refined_start, 3),
                end_sec=round(analysis.refined_end, 3),
                confidence=round(analysis.confidence, 4),
                clip_path=str(clip.clip_path),
                audio_path=str(clip.audio_path) if clip.audio_path else None,
                backend=analysis.match.backend if analysis.match else "none",
            )
        )

    manifest_base = output_dirs["manifests"] / f"{source.video_id}_manifest"
    json_path, csv_path = write_manifests(records, manifest_base)

    logger.info("Pipeline finished with %s clips", len(records))
    logger.info("Manifest JSON: %s", json_path)
    logger.info("Manifest CSV: %s", csv_path)
    _maybe_upload_outputs(config, run_output_root, logger)
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

    segments = detect_music_segments(
        audio_path=working_audio,
        min_segment_sec=config.min_segment,
        max_segment_sec=config.max_segment,
        merge_gap_sec=config.merge_gap,
        expected_song_count=config.expected_song_count,
        logger=logger,
        merge_max_segment_sec=config.merge_max_segment,
        exclude_start_seconds=config.exclude_start_seconds,
        exclude_end_seconds=config.exclude_end_seconds,
    )

    matcher = ChromaprintMatcher(config.ref_library, config.fingerprint_threshold, logger)
    acoustid = AcoustIDClient(config.acoustid_api_key, config.use_acoustid, logger)
    refiner = WhisperXRefiner(device=config.device, logger=logger)

    analyses = _analyze_segments(segments, working_audio, output_dirs, matcher, acoustid, refiner, logger)
    analyses = _split_oversized_analyses(analyses, config.max_segment, logger)
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
