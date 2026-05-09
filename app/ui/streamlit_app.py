"""Streamlit UI for karaoke-clipper."""

from __future__ import annotations

import os
import sys
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
    WHISPERX_BOUNDARY_MODES,
    resolve_runtime_device,
)
from app.integrations.gdrive import find_client_secrets_path
from app.main import preview_pipeline, run_pipeline


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
    gdrive_upload: bool,
    gdrive_folder_id: str,
    gdrive_client_secrets: str,
    gdrive_token: str,
    gdrive_include_tmp: bool,
    gdrive_upload_mode: str,
) -> AppConfig:
    url = url_value.strip() if source_mode == "YouTube URL" else None
    file_path = file_value.strip() if source_mode == "Local file" else None

    profile_values = PROFILES.get(profile, {}) if profile != "custom" else {}
    effective_min_segment = float(profile_values.get("min_segment", min_segment))
    effective_max_segment = float(profile_values.get("max_segment", max_segment))
    effective_merge_gap = float(profile_values.get("merge_gap", merge_gap))
    effective_bridge_noise_gap = float(profile_values.get("bridge_noise_gap", bridge_noise_gap_sec))
    effective_bridge_speech_gap = float(profile_values.get("bridge_speech_gap", bridge_speech_gap_sec))
    effective_pre_roll = float(profile_values.get("pre_roll", pre_roll_sec))
    effective_post_roll = float(profile_values.get("post_roll", post_roll_sec))
    effective_whisperx_mode = str(profile_values.get("whisperx_boundary_mode", whisperx_boundary_mode))
    effective_device = resolve_runtime_device(device)

    return AppConfig(
        url=url or None,
        file=Path(file_path).expanduser().resolve() if file_path else None,
        outdir=Path(outdir_value).expanduser().resolve(),
        audio_clips=audio_clips,
        min_segment=effective_min_segment,
        max_segment=float(effective_max_segment),
        merge_max_segment=float(effective_max_segment),
        segment_tolerance=float(segment_tolerance),
        pre_roll_sec=effective_pre_roll,
        post_roll_sec=effective_post_roll,
        bridge_noise_gap_sec=effective_bridge_noise_gap,
        bridge_speech_gap_sec=effective_bridge_speech_gap,
        use_acoustid=use_acoustid,
        ref_library=Path(ref_library).expanduser().resolve() if ref_library else None,
        device=effective_device,
        sample_rate=int(sample_rate),
        merge_gap=float(effective_merge_gap),
        exclude_start_seconds=float(exclude_start_seconds),
        exclude_end_seconds=float(exclude_end_seconds),
        expected_song_count=expected_song_count,
        clip_mode=clip_mode,
        clip_resolution=clip_resolution,
        fingerprint_threshold=float(fingerprint_threshold),
        acoustid_api_key=os.getenv("ACOUSTID_API_KEY"),
        whisperx_boundary_mode=effective_whisperx_mode,
        whisperx_max_start_shrink_sec=float(whisperx_max_start_shrink),
        whisperx_max_end_shrink_sec=float(whisperx_max_end_shrink),
        allow_hard_split=bool(allow_hard_split),
        energy_frame_ms=int(energy_frame_ms),
        energy_min_active_ms=int(energy_min_active_ms),
        energy_min_silence_ms=int(energy_min_silence_ms),
        profile=str(profile),
        review_score_threshold=float(review_score_threshold),
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


st.set_page_config(page_title="Karaoke Clipper", layout="wide")

st.title("Karaoke Clipper")
st.caption("Preview segments and run the clipper with optional Google Drive upload.")

forced_device_value = os.getenv(RUNTIME_DEVICE_ENV, "").strip().lower()
device_is_forced = forced_device_value in {"cpu", "cuda"}
default_device = forced_device_value if device_is_forced else "cpu"
if device_is_forced:
    st.info(f"Runtime device is forced to '{forced_device_value}' by container healthcheck.")

source_mode = st.radio("Source", ["YouTube URL", "Local file"], horizontal=True)
profile = st.selectbox("Profile", PROFILE_CHOICES, index=0)
profile_defaults = PROFILES.get(profile, {}) if profile != "custom" else {}
if profile != "custom":
    st.info("Profile overrides segment tuning values unless you switch to 'custom'.")

if source_mode == "YouTube URL":
    url_value = st.text_input("YouTube URL", value="")
    file_value = ""
else:
    file_value = st.text_input("Local MP4 path", value="")
    url_value = ""

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    outdir_value = st.text_input("Output parent folder", value="output")
    min_segment = st.number_input(
        "Min segment (sec)",
        min_value=1.0,
        value=float(profile_defaults.get("min_segment", 8.0)),
        step=1.0,
    )
    max_segment = st.number_input(
        "Max segment (sec)",
        min_value=1.0,
        value=float(profile_defaults.get("max_segment", 240.0)),
        step=1.0,
    )
    segment_tolerance = st.number_input(
        "Segment tolerance (sec)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        help="Allow segments to be +/- this many seconds when merging or splitting.",
    )
    pre_roll_sec = st.number_input(
        "Pre-roll (sec)",
        min_value=0.0,
        value=float(profile_defaults.get("pre_roll", 0.5)),
        step=0.1,
    )
    post_roll_sec = st.number_input(
        "Post-roll (sec)",
        min_value=0.0,
        value=float(profile_defaults.get("post_roll", 2.0)),
        step=0.1,
    )
    bridge_noise_gap_sec = st.number_input(
        "Bridge noise gap (sec)",
        min_value=0.0,
        value=float(profile_defaults.get("bridge_noise_gap", 2.0)),
        step=0.1,
    )
    bridge_speech_gap_sec = st.number_input(
        "Bridge speech gap (sec)",
        min_value=0.0,
        value=float(profile_defaults.get("bridge_speech_gap", 1.0)),
        step=0.1,
    )
    merge_gap = st.number_input(
        "Merge gap (sec)",
        min_value=0.0,
        value=float(profile_defaults.get("merge_gap", 2.0)),
        step=0.1,
    )
    exclude_start_seconds = st.number_input("Exclude start (sec)", min_value=0.0, value=0.0, step=0.5)
    exclude_end_seconds = st.number_input("Exclude end (sec)", min_value=0.0, value=0.0, step=0.5)

with col2:
    clip_mode = st.selectbox("Clip mode", ["accurate", "fast"], index=0)
    clip_resolution = st.selectbox("Clip resolution", CLIP_RESOLUTION_CHOICES, index=0)
    audio_clips = st.checkbox("Export WAV clips", value=False)
    use_acoustid = st.checkbox("Use AcoustID lookup", value=False)
    whisperx_boundary_mode = st.selectbox("WhisperX boundary mode", WHISPERX_BOUNDARY_MODES, index=2)
    whisperx_max_start_shrink = st.number_input(
        "WhisperX max start shrink (sec)",
        min_value=0.0,
        value=0.5,
        step=0.1,
    )
    whisperx_max_end_shrink = st.number_input(
        "WhisperX max end shrink (sec)",
        min_value=0.0,
        value=0.5,
        step=0.1,
    )
    allow_hard_split = st.checkbox("Allow hard split", value=False)

with col3:
    sample_rate = st.number_input("Sample rate", min_value=8000, value=16000, step=1000)
    device = st.selectbox(
        "Device",
        ["cpu", "cuda"],
        index=0 if default_device == "cpu" else 1,
        disabled=device_is_forced,
        help="When running in GPU container mode, this value is auto-forced by runtime healthcheck.",
    )
    fingerprint_threshold = st.number_input("Fingerprint threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    review_score_threshold = st.number_input(
        "Review score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.01,
    )
    energy_frame_ms = st.number_input("Energy frame (ms)", min_value=10, value=100, step=10)
    energy_min_active_ms = st.number_input("Energy min active (ms)", min_value=100, value=500, step=50)
    energy_min_silence_ms = st.number_input("Energy min silence (ms)", min_value=100, value=1200, step=50)

expected_enabled = st.checkbox("Use expected song count", value=False)
expected_song_count = None
if expected_enabled:
    expected_song_count = int(st.number_input("Expected song count", min_value=1, value=16, step=1))

ref_library = st.text_input("Reference library JSON", value="data/reference_library.json")

# Default Drive inputs (rendered after preview when user opts-in)
default_folder_id = os.getenv("GDRIVE_FOLDER_ID", "")

client_secrets_default = "secret"
secrets_path = find_client_secrets_path(None)
if secrets_path:
    client_secrets_default = str(secrets_path)

# These variables are placeholders until the user enables Drive upload after preview.
gdrive_folder_id = default_folder_id
gdrive_client_secrets = client_secrets_default
gdrive_token = os.getenv("GDRIVE_TOKEN", "secret/token.json")
gdrive_include_tmp = False
gdrive_upload_mode = "clips"


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


col_preview = st.container()
with col_preview:
    preview_clicked = st.button("Preview segments", width="stretch")

preview_snapshots = st.checkbox("Generate timestamp screenshots", value=False)
snapshot_limit = 0
if preview_snapshots:
    snapshot_limit = int(st.number_input("Screenshot limit", min_value=1, value=12, step=1))

if preview_clicked:
    error = _validate_inputs(source_mode, url_value, file_value)
    if error:
        st.error(error)
    else:
        config = _build_config(
            source_mode=source_mode,
            profile=profile,
            url_value=url_value,
            file_value=file_value,
            outdir_value=outdir_value,
            audio_clips=audio_clips,
            min_segment=min_segment,
            max_segment=max_segment,
            segment_tolerance=segment_tolerance,
            pre_roll_sec=pre_roll_sec,
            post_roll_sec=post_roll_sec,
            bridge_noise_gap_sec=bridge_noise_gap_sec,
            bridge_speech_gap_sec=bridge_speech_gap_sec,
            use_acoustid=use_acoustid,
            ref_library=ref_library,
            device=device,
            sample_rate=sample_rate,
            merge_gap=merge_gap,
            exclude_start_seconds=exclude_start_seconds,
            exclude_end_seconds=exclude_end_seconds,
            expected_song_count=expected_song_count,
            clip_mode=clip_mode,
            clip_resolution=clip_resolution,
            fingerprint_threshold=fingerprint_threshold,
            whisperx_boundary_mode=whisperx_boundary_mode,
            whisperx_max_start_shrink=whisperx_max_start_shrink,
            whisperx_max_end_shrink=whisperx_max_end_shrink,
            allow_hard_split=allow_hard_split,
            energy_frame_ms=energy_frame_ms,
            energy_min_active_ms=energy_min_active_ms,
            energy_min_silence_ms=energy_min_silence_ms,
            review_score_threshold=review_score_threshold,
            gdrive_upload=False,
            gdrive_folder_id=_extract_drive_folder_id(gdrive_folder_id),
            gdrive_client_secrets=gdrive_client_secrets,
            gdrive_token=gdrive_token,
            gdrive_include_tmp=gdrive_include_tmp,
            gdrive_upload_mode=gdrive_upload_mode,
        )

        with st.spinner("Running preview..."):
            try:
                preview_result = preview_pipeline(config, snapshot_limit=snapshot_limit if preview_snapshots else 0)
                st.session_state["preview_result"] = preview_result
                st.session_state["preview_records"] = [record.to_row() for record in preview_result.records]
                st.session_state["preview_source_mode"] = source_mode
                st.session_state["preview_source_url"] = url_value
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Preview failed: {exc}")

if "preview_records" in st.session_state and st.session_state["preview_records"]:
    st.subheader("Preview results")
    st.dataframe(st.session_state["preview_records"], width="stretch")

    preview_result = st.session_state.get("preview_result")
    if preview_result:
        st.subheader("Segment video preview")
        segment_options = [record.index for record in preview_result.records]
        selected_index = st.selectbox("Segment", segment_options, index=0)
        selected_record = preview_result.records[selected_index - 1]

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

    st.markdown("---")
    st.subheader("Run options")
    enable_gdrive_after_preview = st.checkbox(
        "Enable Google Drive upload for full run",
        value=False,
        help="Enable Drive upload only after you've reviewed the preview snapshots",
    )
    if enable_gdrive_after_preview:
        gdrive_folder_id = st.text_input("Drive folder ID or URL", value=st.session_state.get("gdrive_folder_id", default_folder_id))
        gdrive_client_secrets = st.text_input("Client secrets JSON", value=st.session_state.get("gdrive_client_secrets", client_secrets_default))
        gdrive_token = st.text_input("Token cache path", value=st.session_state.get("gdrive_token", os.getenv("GDRIVE_TOKEN", "secret/token.json")))
        gdrive_include_tmp = st.checkbox("Include tmp folder", value=st.session_state.get("gdrive_include_tmp", False))
        gdrive_upload_mode = st.selectbox(
            "Upload mode",
            options=["clips", "all"],
            index=0 if st.session_state.get("gdrive_upload_mode", "clips") == "clips" else 1,
            help="'clips' uploads only the clips folder; 'all' uploads the entire run folder",
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

    st.markdown("")

    st.session_state.setdefault("is_processing", False)
    st.session_state.setdefault("pipeline_done", False)

    placeholder = st.empty()
    if st.session_state["is_processing"]:
        st.markdown("🟡 <span style='color: #FFD700;'>**Processing...**</span>", unsafe_allow_html=True)
        placeholder.progress(0.5)
    elif st.session_state["pipeline_done"]:
        st.markdown("✅ <span style='color: #00AA00;'>**Done**</span>", unsafe_allow_html=True)
        placeholder.empty()
    else:
        run_clicked = st.button("Run full pipeline", width="stretch")
        if run_clicked:
            error = _validate_inputs(source_mode, url_value, file_value)
            if error:
                st.error(error)
            else:
                st.session_state["is_processing"] = True
                st.rerun()

if st.session_state.get("is_processing") and st.session_state.get("preview_records"):
    error = _validate_inputs(source_mode, url_value, file_value)
    if not error:
        gdrive_upload_effective = False
        if st.session_state.get("preview_records"):
            enabled = bool(st.session_state.get("gdrive_folder_id")) and bool(st.session_state.get("gdrive_client_secrets"))
            gdrive_upload_effective = bool(enabled and st.session_state.get("gdrive_folder_id", "").strip())

        if gdrive_upload_effective and not st.session_state.get("gdrive_folder_id", "").strip():
            st.error("Drive folder ID is required when upload is enabled.")
            st.session_state["is_processing"] = False
        else:
            gdrive_folder_id_value = st.session_state.get("gdrive_folder_id", default_folder_id)
            gdrive_client_secrets = st.session_state.get("gdrive_client_secrets", client_secrets_default)
            gdrive_token = st.session_state.get("gdrive_token", os.getenv("GDRIVE_TOKEN", "secret/token.json"))
            gdrive_include_tmp = st.session_state.get("gdrive_include_tmp", False)
            gdrive_upload_mode = st.session_state.get("gdrive_upload_mode", "clips")

            config = _build_config(
                source_mode=source_mode,
                profile=profile,
                url_value=url_value,
                file_value=file_value,
                outdir_value=outdir_value,
                audio_clips=audio_clips,
                min_segment=min_segment,
                max_segment=max_segment,
                segment_tolerance=segment_tolerance,
                pre_roll_sec=pre_roll_sec,
                post_roll_sec=post_roll_sec,
                bridge_noise_gap_sec=bridge_noise_gap_sec,
                bridge_speech_gap_sec=bridge_speech_gap_sec,
                use_acoustid=use_acoustid,
                ref_library=ref_library,
                device=device,
                sample_rate=sample_rate,
                merge_gap=merge_gap,
                exclude_start_seconds=exclude_start_seconds,
                exclude_end_seconds=exclude_end_seconds,
                expected_song_count=expected_song_count,
                clip_mode=clip_mode,
                clip_resolution=clip_resolution,
                fingerprint_threshold=fingerprint_threshold,
                whisperx_boundary_mode=whisperx_boundary_mode,
                whisperx_max_start_shrink=whisperx_max_start_shrink,
                whisperx_max_end_shrink=whisperx_max_end_shrink,
                allow_hard_split=allow_hard_split,
                energy_frame_ms=energy_frame_ms,
                energy_min_active_ms=energy_min_active_ms,
                energy_min_silence_ms=energy_min_silence_ms,
                review_score_threshold=review_score_threshold,
                gdrive_upload=gdrive_upload_effective,
                gdrive_folder_id=_extract_drive_folder_id(gdrive_folder_id_value),
                gdrive_client_secrets=gdrive_client_secrets,
                gdrive_token=gdrive_token,
                gdrive_include_tmp=gdrive_include_tmp,
                gdrive_upload_mode=gdrive_upload_mode,
            )

            export_status = st.empty()
            export_progress = st.empty()

            def _on_export_progress(current: int, total: int, start_sec: float, end_sec: float) -> None:
                if total <= 0:
                    return
                export_progress.progress(min(current / total, 1.0))
                export_status.info(
                    f"Exporting clip {current}/{total}: {start_sec:.2f} -> {end_sec:.2f}"
                )

            with st.spinner("Running pipeline..."):
                try:
                    result = run_pipeline(config, progress_callback=_on_export_progress)
                    if result == 0:
                        st.success("Pipeline finished.")
                        st.session_state["pipeline_done"] = True
                    else:
                        st.warning("Pipeline finished with a non-zero code.")
                        st.session_state["pipeline_done"] = True
                except Exception as exc:  # pragma: no cover - UI error handling
                    st.error(f"Pipeline failed: {exc}")
                    st.session_state["pipeline_done"] = True
                finally:
                    export_status.empty()
                    export_progress.empty()
                    st.session_state["is_processing"] = False
