"""Streamlit UI for karaoke-clipper."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.config import (
    AppConfig,
    CLIP_RESOLUTION_CHOICES,
    PROFILE_CHOICES,
    PROFILES,
    RUNTIME_DEVICE_ENV,
    SINGING_MODEL_MODES,
    WHISPERX_BOUNDARY_MODES,
    resolve_default_singing_model,
    resolve_runtime_device,
)
from app.docker_gpu_runner import DockerGpuRunnerError, run_pipeline_in_docker
from app.integrations.gdrive import find_client_secrets_path
from app.main import preview_pipeline, run_pipeline


def _upload_outputs_from_host(config: AppConfig, output_root: Path) -> str:
    if not config.gdrive_folder_id:
        raise ValueError("Drive folder ID is required when upload is enabled.")

    if config.gdrive_upload_mode == "all":
        from app.integrations.gdrive import upload_output_dir
        from app.utils.logging import setup_logger

        logger = setup_logger(output_root / "logs" / "gdrive_upload_host.log", name="karaoke_clipper_gdrive_host")
        return upload_output_dir(
            output_dir=output_root,
            parent_folder_id=config.gdrive_folder_id,
            client_secrets_path=config.gdrive_client_secrets,
            token_path=config.gdrive_token_path,
            include_tmp=config.gdrive_include_tmp,
            logger=logger,
        )

    from app.integrations.gdrive import upload_clips_dir
    from app.utils.logging import setup_logger

    logger = setup_logger(output_root / "logs" / "gdrive_upload_host.log", name="karaoke_clipper_gdrive_host")
    return upload_clips_dir(
        output_dir=output_root,
        parent_folder_id=config.gdrive_folder_id,
        client_secrets_path=config.gdrive_client_secrets,
        token_path=config.gdrive_token_path,
        logger=logger,
    )


def _build_config(
    source_mode: str,
    profile: str,
    url_value: str,
    file_value: str,
    outdir_value: str,
    audio_clips: bool,
    min_segment: float,
    max_segment: float,
    segment_tolerance: float,
    pre_roll_sec: float,
    post_roll_sec: float,
    bridge_noise_gap_sec: float,
    bridge_speech_gap_sec: float,
    use_acoustid: bool,
    ref_library: str,
    device: str,
    sample_rate: int,
    merge_gap: float,
    exclude_start_seconds: float,
    exclude_end_seconds: float,
    expected_song_count: int | None,
    clip_mode: str,
    clip_resolution: str,
    fingerprint_threshold: float,
    whisperx_boundary_mode: str,
    whisperx_max_start_shrink: float,
    whisperx_max_end_shrink: float,
    allow_hard_split: bool,
    energy_frame_ms: int,
    energy_min_active_ms: int,
    energy_min_silence_ms: int,
    review_score_threshold: float,
    music_ratio_threshold: float,
    singing_model_path: str,
    singing_score_threshold: float,
    singing_model_mode: str,
    gdrive_upload: bool,
    gdrive_folder_id: str,
    gdrive_client_secrets: str,
    gdrive_token: str,
    gdrive_include_tmp: bool,
    gdrive_upload_mode: str,
) -> AppConfig:
    url = url_value.strip() if source_mode == "YouTube URL" else None
    file_path = file_value.strip() if source_mode == "Local file" else None
    effective_device = resolve_runtime_device(device)

    return AppConfig(
        url=url or None,
        file=Path(file_path).expanduser().resolve() if file_path else None,
        outdir=Path(outdir_value).expanduser().resolve(),
        audio_clips=audio_clips,
        min_segment=float(min_segment),
        max_segment=float(max_segment),
        merge_max_segment=float(max_segment),
        segment_tolerance=float(segment_tolerance),
        pre_roll_sec=float(pre_roll_sec),
        post_roll_sec=float(post_roll_sec),
        bridge_noise_gap_sec=float(bridge_noise_gap_sec),
        bridge_speech_gap_sec=float(bridge_speech_gap_sec),
        use_acoustid=use_acoustid,
        ref_library=Path(ref_library).expanduser().resolve() if ref_library else None,
        device=effective_device,
        sample_rate=int(sample_rate),
        merge_gap=float(merge_gap),
        exclude_start_seconds=float(exclude_start_seconds),
        exclude_end_seconds=float(exclude_end_seconds),
        expected_song_count=expected_song_count,
        clip_mode=clip_mode,
        clip_resolution=clip_resolution,
        fingerprint_threshold=float(fingerprint_threshold),
        acoustid_api_key=os.getenv("ACOUSTID_API_KEY"),
        whisperx_boundary_mode=str(whisperx_boundary_mode),
        whisperx_max_start_shrink_sec=float(whisperx_max_start_shrink),
        whisperx_max_end_shrink_sec=float(whisperx_max_end_shrink),
        allow_hard_split=bool(allow_hard_split),
        energy_frame_ms=int(energy_frame_ms),
        energy_min_active_ms=int(energy_min_active_ms),
        energy_min_silence_ms=int(energy_min_silence_ms),
        profile=str(profile),
        review_score_threshold=float(review_score_threshold),
        music_ratio_threshold=float(music_ratio_threshold),
        singing_model_path=Path(singing_model_path).expanduser().resolve() if singing_model_path else None,
        singing_score_threshold=float(singing_score_threshold),
        singing_model_mode=str(singing_model_mode),
        gdrive_upload=gdrive_upload,
        gdrive_folder_id=(gdrive_folder_id or "").strip() or os.getenv("GDRIVE_FOLDER_ID"),
        gdrive_client_secrets=(
            Path(gdrive_client_secrets).expanduser().resolve()
            if gdrive_client_secrets
            else None
        ),
        gdrive_token_path=Path(gdrive_token).expanduser().resolve(),
        gdrive_include_tmp=gdrive_include_tmp,
        gdrive_upload_mode=gdrive_upload_mode,
    )


def _validate_inputs(source_mode: str, url_value: str, file_value: str) -> str | None:
    if source_mode == "YouTube URL" and not url_value.strip():
        return "Please enter a YouTube URL."
    if source_mode == "Local file":
        if not file_value.strip():
            return "Please enter a local file path."
        if not Path(file_value).expanduser().exists():
            return "Local file path does not exist."
    return None


def _extract_drive_folder_id(value: str | None) -> str | None:
    if not value:
        return None
    val = value.strip()
    if "drive.google.com" in val:
        if "/folders/" in val:
            parts = val.split("/folders/")
            if len(parts) > 1:
                return parts[1].split("?")[0].strip("/ ")
        if "id=" in val:
            for part in val.split("&"):
                if part.startswith("id="):
                    return part.split("=", 1)[1]
    return val or None


def _relative_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _build_config_from_inputs(
    values: dict[str, object],
    *,
    gdrive_upload: bool,
    gdrive_folder_id: str | None,
    gdrive_client_secrets: str,
    gdrive_token: str,
    gdrive_include_tmp: bool,
    gdrive_upload_mode: str,
) -> AppConfig:
    expected_song_count = values.get("expected_song_count")
    return _build_config(
        source_mode=str(values["source_mode"]),
        profile=str(values["profile"]),
        url_value=str(values["url_value"]),
        file_value=str(values["file_value"]),
        outdir_value=str(values["outdir_value"]),
        audio_clips=bool(values["audio_clips"]),
        min_segment=float(values["min_segment"]),
        max_segment=float(values["max_segment"]),
        segment_tolerance=float(values["segment_tolerance"]),
        pre_roll_sec=float(values["pre_roll_sec"]),
        post_roll_sec=float(values["post_roll_sec"]),
        bridge_noise_gap_sec=float(values["bridge_noise_gap_sec"]),
        bridge_speech_gap_sec=float(values["bridge_speech_gap_sec"]),
        use_acoustid=bool(values["use_acoustid"]),
        ref_library=str(values["ref_library"]),
        device=str(values["device"]),
        sample_rate=int(values["sample_rate"]),
        merge_gap=float(values["merge_gap"]),
        exclude_start_seconds=float(values["exclude_start_seconds"]),
        exclude_end_seconds=float(values["exclude_end_seconds"]),
        expected_song_count=expected_song_count if type(expected_song_count) is int else None,
        clip_mode=str(values["clip_mode"]),
        clip_resolution=str(values["clip_resolution"]),
        fingerprint_threshold=float(values["fingerprint_threshold"]),
        whisperx_boundary_mode=str(values["whisperx_boundary_mode"]),
        whisperx_max_start_shrink=float(values["whisperx_max_start_shrink"]),
        whisperx_max_end_shrink=float(values["whisperx_max_end_shrink"]),
        allow_hard_split=bool(values["allow_hard_split"]),
        energy_frame_ms=int(values["energy_frame_ms"]),
        energy_min_active_ms=int(values["energy_min_active_ms"]),
        energy_min_silence_ms=int(values["energy_min_silence_ms"]),
        review_score_threshold=float(values["review_score_threshold"]),
        music_ratio_threshold=float(values["music_ratio_threshold"]),
        singing_model_path=str(values["singing_model_path"]),
        singing_score_threshold=float(values["singing_score_threshold"]),
        singing_model_mode=str(values["singing_model_mode"]),
        gdrive_upload=gdrive_upload,
        gdrive_folder_id=gdrive_folder_id or "",
        gdrive_client_secrets=gdrive_client_secrets,
        gdrive_token=gdrive_token,
        gdrive_include_tmp=gdrive_include_tmp,
        gdrive_upload_mode=gdrive_upload_mode,
    )


def _render_source_setup(
    default_singing_model_path: Path | None,
    forced_device_value: str,
    device_is_forced: bool,
) -> tuple[dict[str, object], dict[str, float | str]]:
    values: dict[str, object] = {}
    with st.expander("Source and run setup", expanded=True):
        status_parts: list[str] = []
        if device_is_forced:
            status_parts.append(f"Runtime device forced to {forced_device_value}")
        if default_singing_model_path is not None:
            status_parts.append(f"Default singing model: {_relative_display_path(default_singing_model_path)}")
        if status_parts:
            st.caption(" | ".join(status_parts))

        values["source_mode"] = st.radio("Source", ["YouTube URL", "Local file"], horizontal=True)

        col_profile, col_output = st.columns(2)
        with col_profile:
            profile = st.selectbox("Profile", PROFILE_CHOICES, index=0)
        with col_output:
            values["outdir_value"] = st.text_input("Output parent folder", value="output")

        values["profile"] = profile
        profile_defaults = PROFILES.get(profile, {}) if profile != "custom" else {}
        if profile != "custom":
            st.caption("Profile defaults are applied to the tuning controls below.")

        if values["source_mode"] == "YouTube URL":
            values["url_value"] = st.text_input("YouTube URL", value="")
            values["file_value"] = ""
        else:
            values["file_value"] = st.text_input("Local MP4 path", value="")
            values["url_value"] = ""

    return values, profile_defaults


def _render_clip_settings(values: dict[str, object], profile_defaults: dict[str, float | str]) -> None:
    with st.expander("Clip settings", expanded=False):
        col_timing, col_export = st.columns(2)
        with col_timing:
            values["min_segment"] = st.number_input(
                "Min segment (sec)",
                min_value=1.0,
                value=float(profile_defaults.get("min_segment", 8.0)),
                step=1.0,
            )
            values["max_segment"] = st.number_input(
                "Max segment (sec)",
                min_value=1.0,
                value=float(profile_defaults.get("max_segment", 240.0)),
                step=1.0,
            )
            values["segment_tolerance"] = st.number_input(
                "Segment tolerance (sec)",
                min_value=0.0,
                value=0.0,
                step=0.5,
                help="Allow segments to be +/- this many seconds when merging or splitting.",
            )
            values["pre_roll_sec"] = st.number_input(
                "Pre-roll (sec)",
                min_value=0.0,
                value=float(profile_defaults.get("pre_roll", 0.5)),
                step=0.1,
            )
            values["post_roll_sec"] = st.number_input(
                "Post-roll (sec)",
                min_value=0.0,
                value=float(profile_defaults.get("post_roll", 2.0)),
                step=0.1,
            )
        with col_export:
            values["clip_mode"] = st.selectbox(
                "Clip mode",
                ["accurate", "fast"],
                index=0,
                help=(
                    "Accurate mode re-encodes clips and uses NVIDIA NVENC when the runtime device is cuda. "
                    "Fast mode with source resolution uses stream copy, so GPU usage may stay near zero."
                ),
            )
            values["clip_resolution"] = st.selectbox("Clip resolution", CLIP_RESOLUTION_CHOICES, index=0)
            values["audio_clips"] = st.checkbox("Export WAV clips", value=False)
            values["exclude_start_seconds"] = st.number_input("Exclude start (sec)", min_value=0.0, value=0.0, step=0.5)
            values["exclude_end_seconds"] = st.number_input("Exclude end (sec)", min_value=0.0, value=0.0, step=0.5)


def _render_detection_settings(values: dict[str, object], profile_defaults: dict[str, float | str]) -> None:
    with st.expander("Detection settings", expanded=False):
        col_merge, col_refine = st.columns(2)
        with col_merge:
            values["bridge_noise_gap_sec"] = st.number_input(
                "Bridge noise gap (sec)",
                min_value=0.0,
                value=float(profile_defaults.get("bridge_noise_gap", 2.0)),
                step=0.1,
            )
            values["bridge_speech_gap_sec"] = st.number_input(
                "Bridge speech gap (sec)",
                min_value=0.0,
                value=float(profile_defaults.get("bridge_speech_gap", 1.0)),
                step=0.1,
            )
            values["merge_gap"] = st.number_input(
                "Merge gap (sec)",
                min_value=0.0,
                value=float(profile_defaults.get("merge_gap", 2.0)),
                step=0.1,
            )
            expected_enabled = st.checkbox("Use expected song count", value=False)
            values["expected_song_count"] = None
            if expected_enabled:
                values["expected_song_count"] = int(st.number_input("Expected song count", min_value=1, value=16, step=1))
        with col_refine:
            values["use_acoustid"] = st.checkbox("Use AcoustID lookup", value=False)
            default_whisper_mode = str(profile_defaults.get("whisperx_boundary_mode", "safe"))
            default_whisper_index = (
                WHISPERX_BOUNDARY_MODES.index(default_whisper_mode)
                if default_whisper_mode in WHISPERX_BOUNDARY_MODES
                else 2
            )
            values["whisperx_boundary_mode"] = st.selectbox(
                "WhisperX boundary mode",
                WHISPERX_BOUNDARY_MODES,
                index=default_whisper_index,
            )
            values["whisperx_max_start_shrink"] = st.number_input(
                "WhisperX max start shrink (sec)",
                min_value=0.0,
                value=0.5,
                step=0.1,
            )
            values["whisperx_max_end_shrink"] = st.number_input(
                "WhisperX max end shrink (sec)",
                min_value=0.0,
                value=0.5,
                step=0.1,
            )
            values["allow_hard_split"] = st.checkbox("Allow hard split", value=False)


def _render_model_scoring(
    values: dict[str, object],
    default_singing_model_path: Path | None,
    default_singing_model_mode: str,
) -> None:
    with st.expander("Model scoring", expanded=False):
        col_thresholds, col_model = st.columns(2)
        with col_thresholds:
            values["fingerprint_threshold"] = st.number_input(
                "Fingerprint threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.01,
            )
            values["review_score_threshold"] = st.number_input(
                "Review score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.65,
                step=0.01,
            )
            values["music_ratio_threshold"] = st.number_input(
                "Music ratio threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Filter out segments whose music_ratio is lower than this value.",
            )
        with col_model:
            default_singing_mode_index = (
                SINGING_MODEL_MODES.index(default_singing_model_mode)
                if default_singing_model_mode in SINGING_MODEL_MODES
                else 0
            )
            values["singing_model_mode"] = st.selectbox(
                "Singing model mode",
                SINGING_MODEL_MODES,
                index=default_singing_mode_index,
            )
            values["singing_model_path"] = st.text_input(
                "Singing model path",
                value=str(default_singing_model_path) if default_singing_model_path else "",
            )
            values["singing_score_threshold"] = st.number_input(
                "Singing score threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
            )


def _render_runtime_export(
    values: dict[str, object],
    default_device: str,
    device_is_forced: bool,
) -> tuple[bool, int]:
    with st.expander("Runtime and export", expanded=False):
        col_runtime, col_energy = st.columns(2)
        with col_runtime:
            values["sample_rate"] = st.number_input("Sample rate", min_value=8000, value=16000, step=1000)
            values["device"] = st.selectbox(
                "Device",
                ["cpu", "cuda"],
                index=0 if default_device == "cpu" else 1,
                disabled=device_is_forced,
                help="When running in GPU container mode, this value is auto-forced by runtime healthcheck.",
            )
            values["ref_library"] = st.text_input("Reference library JSON", value="data/reference_library.json")
            preview_snapshots = st.checkbox("Generate timestamp screenshots", value=False)
            snapshot_limit = 0
            if preview_snapshots:
                snapshot_limit = int(st.number_input("Screenshot limit", min_value=1, value=12, step=1))
        with col_energy:
            values["energy_frame_ms"] = st.number_input("Energy frame (ms)", min_value=10, value=100, step=10)
            values["energy_min_active_ms"] = st.number_input("Energy min active (ms)", min_value=100, value=500, step=50)
            values["energy_min_silence_ms"] = st.number_input(
                "Energy min silence (ms)",
                min_value=100,
                value=1200,
                step=50,
            )
    return preview_snapshots, snapshot_limit


def _render_preview_action(values: dict[str, object], preview_snapshots: bool, snapshot_limit: int) -> None:
    preview_clicked = st.button("Preview segments", width="stretch")
    if not preview_clicked:
        return

    error = _validate_inputs(str(values["source_mode"]), str(values["url_value"]), str(values["file_value"]))
    if error:
        st.error(error)
        return

    config = _build_config_from_inputs(
        values,
        gdrive_upload=False,
        gdrive_folder_id=_extract_drive_folder_id(st.session_state.get("gdrive_folder_id", "")),
        gdrive_client_secrets=str(st.session_state.get("gdrive_client_secrets", "")),
        gdrive_token=str(st.session_state.get("gdrive_token", os.getenv("GDRIVE_TOKEN", "secret/token.json"))),
        gdrive_include_tmp=bool(st.session_state.get("gdrive_include_tmp", False)),
        gdrive_upload_mode=str(st.session_state.get("gdrive_upload_mode", "clips")),
    )

    with st.spinner("Running preview..."):
        try:
            preview_result = preview_pipeline(config, snapshot_limit=snapshot_limit if preview_snapshots else 0)
            st.session_state["preview_result"] = preview_result
            st.session_state["preview_records"] = [record.to_row() for record in preview_result.records]
            st.session_state["preview_source_mode"] = values["source_mode"]
            st.session_state["preview_source_url"] = values["url_value"]
        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Preview failed: {exc}")


def _render_preview_results(preview_snapshots: bool) -> None:
    if not st.session_state.get("preview_records"):
        return

    st.subheader("Preview results")
    st.dataframe(st.session_state["preview_records"], width="stretch")

    preview_result = st.session_state.get("preview_result")
    if not preview_result:
        return

    st.subheader("Segment video preview")
    segment_options = [record.index for record in preview_result.records]
    selected_index = st.selectbox("Segment", segment_options, index=0)
    selected_record = next(
        (record for record in preview_result.records if record.index == selected_index),
        preview_result.records[0],
    )

    if st.session_state.get("preview_source_mode") == "YouTube URL":
        st.video(st.session_state.get("preview_source_url", ""), start_time=int(selected_record.start_sec))
    else:
        st.video(str(preview_result.source_video), start_time=int(selected_record.start_sec))

    if preview_result.snapshots:
        st.subheader("Timestamp screenshots")
        cols = st.columns(3)
        for idx, snapshot in enumerate(preview_result.snapshots):
            cols[idx % 3].image(str(snapshot), caption=snapshot.name)
    elif preview_snapshots:
        st.info("No screenshots were generated (source may be audio-only).")


def _render_gdrive_options(default_folder_id: str, client_secrets_default: str) -> None:
    enable_gdrive_after_preview = st.checkbox(
        "Enable Google Drive upload for full run",
        value=bool(st.session_state.get("gdrive_upload_enabled", False)),
        help="Enable Drive upload only after you've reviewed the preview snapshots.",
    )
    st.session_state["gdrive_upload_enabled"] = enable_gdrive_after_preview

    if enable_gdrive_after_preview:
        gdrive_folder_id = st.text_input(
            "Drive folder ID or URL",
            value=st.session_state.get("gdrive_folder_id", default_folder_id),
        )
        gdrive_client_secrets = st.text_input(
            "Client secrets JSON",
            value=st.session_state.get("gdrive_client_secrets", client_secrets_default),
        )
        gdrive_token = st.text_input(
            "Token cache path",
            value=st.session_state.get("gdrive_token", os.getenv("GDRIVE_TOKEN", "secret/token.json")),
        )
        gdrive_include_tmp = st.checkbox("Include tmp folder", value=st.session_state.get("gdrive_include_tmp", False))
        gdrive_upload_mode = st.selectbox(
            "Upload mode",
            options=["clips", "all"],
            index=0 if st.session_state.get("gdrive_upload_mode", "clips") == "clips" else 1,
            help="'clips' uploads only the clips folder; 'all' uploads the entire run folder.",
        )
        st.session_state["gdrive_folder_id"] = gdrive_folder_id
        st.session_state["gdrive_client_secrets"] = gdrive_client_secrets
        st.session_state["gdrive_token"] = gdrive_token
        st.session_state["gdrive_include_tmp"] = gdrive_include_tmp
        st.session_state["gdrive_upload_mode"] = gdrive_upload_mode
    else:
        st.session_state.setdefault("gdrive_folder_id", default_folder_id)
        st.session_state.setdefault("gdrive_client_secrets", client_secrets_default)
        st.session_state.setdefault("gdrive_token", os.getenv("GDRIVE_TOKEN", "secret/token.json"))
        st.session_state.setdefault("gdrive_include_tmp", False)
        st.session_state.setdefault("gdrive_upload_mode", "clips")


def _render_run_options(
    values: dict[str, object],
    *,
    default_folder_id: str,
    client_secrets_default: str,
    device_is_forced: bool,
) -> None:
    if not st.session_state.get("preview_records"):
        return

    st.markdown("---")
    st.subheader("Run options")
    with st.expander("Runtime and upload for full run", expanded=True):
        _render_gdrive_options(default_folder_id, client_secrets_default)

        if device_is_forced:
            st.session_state["use_docker_gpu_pipeline"] = False
        else:
            use_docker_gpu_pipeline = st.checkbox(
                "Use Docker GPU for full run",
                value=bool(st.session_state.get("use_docker_gpu_pipeline", True)),
                help="Runs the full pipeline inside karaoke-clipper:gpu with host folders bind-mounted.",
            )
            st.session_state["use_docker_gpu_pipeline"] = use_docker_gpu_pipeline

    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("pipeline_done", False)

    placeholder = st.empty()
    if st.session_state["is_processing"]:
        st.markdown("**Processing...**")
        placeholder.progress(0.5)
    elif st.session_state["pipeline_done"]:
        st.markdown("**Done**")
        placeholder.empty()
    else:
        run_clicked = st.button("Run full pipeline", width="stretch")
        if run_clicked:
            error = _validate_inputs(str(values["source_mode"]), str(values["url_value"]), str(values["file_value"]))
            if error:
                st.error(error)
            else:
                st.session_state["is_processing"] = True
                st.rerun()


def _run_pipeline_if_requested(
    values: dict[str, object],
    *,
    default_folder_id: str,
    client_secrets_default: str,
    device_is_forced: bool,
) -> None:
    if not st.session_state.get("is_processing") or not st.session_state.get("preview_records"):
        return

    error = _validate_inputs(str(values["source_mode"]), str(values["url_value"]), str(values["file_value"]))
    if error:
        st.error(error)
        st.session_state["is_processing"] = False
        return

    gdrive_upload_effective = bool(st.session_state.get("gdrive_upload_enabled", False))
    if gdrive_upload_effective and not str(st.session_state.get("gdrive_folder_id", "")).strip():
        st.error("Drive folder ID is required when upload is enabled.")
        st.session_state["is_processing"] = False
        return

    gdrive_folder_id_value = str(st.session_state.get("gdrive_folder_id", default_folder_id))
    gdrive_client_secrets = str(st.session_state.get("gdrive_client_secrets", client_secrets_default))
    gdrive_token = str(st.session_state.get("gdrive_token", os.getenv("GDRIVE_TOKEN", "secret/token.json")))
    gdrive_include_tmp = bool(st.session_state.get("gdrive_include_tmp", False))
    gdrive_upload_mode = str(st.session_state.get("gdrive_upload_mode", "clips"))

    config = _build_config_from_inputs(
        values,
        gdrive_upload=gdrive_upload_effective,
        gdrive_folder_id=_extract_drive_folder_id(gdrive_folder_id_value),
        gdrive_client_secrets=gdrive_client_secrets,
        gdrive_token=gdrive_token,
        gdrive_include_tmp=gdrive_include_tmp,
        gdrive_upload_mode=gdrive_upload_mode,
    )

    export_status = st.empty()
    export_progress = st.empty()
    docker_log_box = st.empty()
    docker_log_lines: list[str] = []

    def _on_export_progress(current: int, total: int, start_sec: float, end_sec: float) -> None:
        if total <= 0:
            return
        export_progress.progress(min(current / total, 1.0))
        export_status.info(f"Exporting clip {current}/{total}: {start_sec:.2f} -> {end_sec:.2f}")

    def _on_docker_log(message: str) -> None:
        if not message:
            return
        docker_log_lines.append(message)
        del docker_log_lines[:-80]
        docker_log_box.code("\n".join(docker_log_lines), language="text")

    with st.spinner("Running pipeline..."):
        try:
            use_docker_gpu_pipeline = bool(st.session_state.get("use_docker_gpu_pipeline", False)) and not device_is_forced
            if use_docker_gpu_pipeline:
                docker_config = replace(config, gdrive_upload=False)
                result = run_pipeline_in_docker(docker_config, log_callback=_on_docker_log)
                if result == 0 and config.gdrive_upload:
                    preview_result_for_upload = st.session_state.get("preview_result")
                    output_root = (
                        preview_result_for_upload.output_root
                        if preview_result_for_upload is not None
                        else config.outdir
                    )
                    try:
                        export_status.info("Uploading clips to Google Drive from local host...")
                        uploaded_folder_id = _upload_outputs_from_host(config, Path(output_root))
                        export_status.success(f"Google Drive upload finished: {uploaded_folder_id}")
                    except Exception as exc:  # pragma: no cover - network/auth dependent path
                        st.error(f"Google Drive upload failed after pipeline finished: {exc}")
            else:
                result = run_pipeline(config, progress_callback=_on_export_progress)
            if result == 0:
                st.success("Pipeline finished.")
                st.session_state["pipeline_done"] = True
            else:
                st.warning("Pipeline finished with a non-zero code.")
                st.session_state["pipeline_done"] = True
        except DockerGpuRunnerError as exc:  # pragma: no cover - depends on local Docker
            st.error(f"Docker GPU pipeline failed: {exc}")
            st.session_state["pipeline_done"] = True
        except Exception as exc:  # pragma: no cover - UI error handling
            st.error(f"Pipeline failed: {exc}")
            st.session_state["pipeline_done"] = True
        finally:
            export_status.empty()
            export_progress.empty()
            st.session_state["is_processing"] = False


def main() -> None:
    st.set_page_config(page_title="Karaoke Clipper", layout="wide")

    st.title("Karaoke Clipper")
    st.caption("Preview segments and run the clipper with optional Google Drive upload.")

    forced_device_value = os.getenv(RUNTIME_DEVICE_ENV, "").strip().lower()
    device_is_forced = forced_device_value in {"cpu", "cuda"}
    default_device = forced_device_value if device_is_forced else "cpu"
    default_singing_model_path, default_singing_model_mode = resolve_default_singing_model(PROJECT_ROOT)

    default_folder_id = os.getenv("GDRIVE_FOLDER_ID", "")
    client_secrets_default = "secret"
    secrets_path = find_client_secrets_path(None)
    if secrets_path:
        client_secrets_default = str(secrets_path)

    values, profile_defaults = _render_source_setup(
        default_singing_model_path,
        forced_device_value,
        device_is_forced,
    )
    _render_clip_settings(values, profile_defaults)
    _render_detection_settings(values, profile_defaults)
    _render_model_scoring(values, default_singing_model_path, default_singing_model_mode)
    preview_snapshots, snapshot_limit = _render_runtime_export(values, default_device, device_is_forced)

    _render_preview_action(values, preview_snapshots, snapshot_limit)
    _render_preview_results(preview_snapshots)
    _render_run_options(
        values,
        default_folder_id=default_folder_id,
        client_secrets_default=client_secrets_default,
        device_is_forced=device_is_forced,
    )
    _run_pipeline_if_requested(
        values,
        default_folder_id=default_folder_id,
        client_secrets_default=client_secrets_default,
        device_is_forced=device_is_forced,
    )


main()
