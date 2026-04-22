"""CLI entrypoint and end-to-end pipeline orchestration."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

from app.align.whisperx_align import WhisperXRefiner
from app.clip.cutter import export_clip
from app.config import AppConfig, load_config
from app.identify.acoustid_client import AcoustIDClient
from app.identify.chromaprint_match import ChromaprintMatcher
from app.ingest.youtube import SourceVideo, download_youtube_video, register_local_video
from app.output.manifest import ManifestRecord, write_manifests
from app.preprocess.extract_audio import extract_working_audio
from app.segment.music_segments import detect_music_segments
from app.utils.ffmpeg import ensure_ffmpeg_available
from app.utils.logging import setup_logger
from app.utils.paths import prepare_output_dirs
from app.utils.timecode import sanitize_filename_component


def _resolve_source(config: AppConfig, vods_dir: Path, logger) -> SourceVideo:
    if config.url:
        return download_youtube_video(config.url, vods_dir, logger)
    assert config.file is not None
    return register_local_video(config.file)


def run_pipeline(config: AppConfig) -> int:
    output_dirs = prepare_output_dirs(config.outdir)
    log_path = output_dirs["logs"] / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
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
        logger=logger,
    )

    matcher = ChromaprintMatcher(config.ref_library, config.fingerprint_threshold, logger)
    acoustid = AcoustIDClient(config.acoustid_api_key, config.use_acoustid, logger)
    refiner = WhisperXRefiner(device=config.device, logger=logger)

    records: list[ManifestRecord] = []

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

        label = f"song_{idx:03d}"
        if match is not None:
            label = sanitize_filename_component(f"{match.artist} - {match.song}__{idx:03d}")

        clip = export_clip(
            video_path=source.video_path,
            start_sec=refined_start,
            end_sec=refined_end,
            clips_dir=output_dirs["clips"],
            clip_stem=label,
            include_audio_clip=config.audio_clips,
            mode=config.clip_mode,
            logger=logger,
        )

        records.append(
            ManifestRecord(
                source_video=str(source.video_path),
                video_id=source.video_id,
                song=match.song if match else "Unknown",
                artist=match.artist if match else "Unknown",
                start_sec=round(refined_start, 3),
                end_sec=round(refined_end, 3),
                confidence=round(match.confidence if match else segment.score, 4),
                clip_path=str(clip.clip_path),
                audio_path=str(clip.audio_path) if clip.audio_path else None,
                backend=match.backend if match else "none",
            )
        )

    manifest_base = output_dirs["manifests"] / f"{source.video_id}_manifest"
    json_path, csv_path = write_manifests(records, manifest_base)

    logger.info("Pipeline finished with %s clips", len(records))
    logger.info("Manifest JSON: %s", json_path)
    logger.info("Manifest CSV: %s", csv_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    config = load_config(argv)
    return run_pipeline(config)


if __name__ == "__main__":
    raise SystemExit(main())
