"""Streamlit UI for karaoke-clipper."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.config import AppConfig, CLIP_RESOLUTION_CHOICES
from app.integrations.gdrive import find_client_secrets_path
from app.main import preview_pipeline, run_pipeline


def _build_config(
    source_mode: str,
    url_value: str,
    file_value: str,
    outdir_value: str,
    audio_clips: bool,
    min_segment: float,
    max_segment: float,
    use_acoustid: bool,
    ref_library: str,
    device: str,
    sample_rate: int,
    merge_gap: float,
    exclude_start_seconds: float,
    expected_song_count: int | None,
    clip_mode: str,
    clip_resolution: str,
    fingerprint_threshold: float,
    gdrive_upload: bool,
    gdrive_folder_id: str,
    gdrive_client_secrets: str,
    gdrive_token: str,
    gdrive_include_tmp: bool,
) -> AppConfig:
    url = url_value.strip() if source_mode == "YouTube URL" else None
    file_path = file_value.strip() if source_mode == "Local file" else None

    return AppConfig(
        url=url or None,
        file=Path(file_path).expanduser().resolve() if file_path else None,
        outdir=Path(outdir_value).expanduser().resolve(),
        audio_clips=audio_clips,
        min_segment=float(min_segment),
        max_segment=float(max_segment),
        use_acoustid=use_acoustid,
        ref_library=Path(ref_library).expanduser().resolve() if ref_library else None,
        device=device,
        sample_rate=int(sample_rate),
        merge_gap=float(merge_gap),
        exclude_start_seconds=float(exclude_start_seconds),
        expected_song_count=expected_song_count,
        clip_mode=clip_mode,
        clip_resolution=clip_resolution,
        fingerprint_threshold=float(fingerprint_threshold),
        acoustid_api_key=os.getenv("ACOUSTID_API_KEY"),
        gdrive_upload=gdrive_upload,
        gdrive_folder_id=gdrive_folder_id.strip() or os.getenv("GDRIVE_FOLDER_ID"),
        gdrive_client_secrets=(
            Path(gdrive_client_secrets).expanduser().resolve()
            if gdrive_client_secrets
            else None
        ),
        gdrive_token_path=Path(gdrive_token).expanduser().resolve(),
        gdrive_include_tmp=gdrive_include_tmp,
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

source_mode = st.radio("Source", ["YouTube URL", "Local file"], horizontal=True)

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
    min_segment = st.number_input("Min segment (sec)", min_value=1.0, value=8.0, step=1.0)
    max_segment = st.number_input("Max segment (sec)", min_value=1.0, value=240.0, step=1.0)
    merge_gap = st.number_input("Merge gap (sec)", min_value=0.0, value=2.0, step=0.1)
    exclude_start_seconds = st.number_input("Exclude start (sec)", min_value=0.0, value=0.0, step=0.5)

with col2:
    clip_mode = st.selectbox("Clip mode", ["accurate", "fast"], index=0)
    clip_resolution = st.selectbox("Clip resolution", CLIP_RESOLUTION_CHOICES, index=0)
    audio_clips = st.checkbox("Export WAV clips", value=False)
    use_acoustid = st.checkbox("Use AcoustID lookup", value=False)

with col3:
    sample_rate = st.number_input("Sample rate", min_value=8000, value=16000, step=1000)
    device = st.selectbox("Device", ["cpu", "cuda"], index=0)
    fingerprint_threshold = st.number_input("Fingerprint threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

expected_enabled = st.checkbox("Use expected song count", value=False)
expected_song_count = None
if expected_enabled:
    expected_song_count = int(st.number_input("Expected song count", min_value=1, value=16, step=1))

ref_library = st.text_input("Reference library JSON", value="data/reference_library.json")

st.markdown("---")

st.subheader("Google Drive Upload")

gdrive_upload = st.checkbox("Upload output to Google Drive", value=False)

default_folder_id = os.getenv("GDRIVE_FOLDER_ID", "")

client_secrets_default = "secret"
secrets_path = find_client_secrets_path(None)
if secrets_path:
    client_secrets_default = str(secrets_path)

if gdrive_upload:
    gdrive_folder_id = st.text_input("Drive folder ID", value=default_folder_id)
    gdrive_client_secrets = st.text_input("Client secrets JSON", value=client_secrets_default)
    gdrive_token = st.text_input("Token cache path", value=os.getenv("GDRIVE_TOKEN", "secret/token.json"))
    gdrive_include_tmp = st.checkbox("Include tmp folder", value=False)
else:
    gdrive_folder_id = default_folder_id
    gdrive_client_secrets = client_secrets_default
    gdrive_token = os.getenv("GDRIVE_TOKEN", "secret/token.json")
    gdrive_include_tmp = False

st.markdown("---")

col_preview, col_run = st.columns(2)

with col_preview:
    preview_clicked = st.button("Preview segments", width="stretch")

with col_run:
    run_clicked = st.button("Run full pipeline", width="stretch")

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
            url_value=url_value,
            file_value=file_value,
            outdir_value=outdir_value,
            audio_clips=audio_clips,
            min_segment=min_segment,
            max_segment=max_segment,
            use_acoustid=use_acoustid,
            ref_library=ref_library,
            device=device,
            sample_rate=sample_rate,
            merge_gap=merge_gap,
            exclude_start_seconds=exclude_start_seconds,
            expected_song_count=expected_song_count,
            clip_mode=clip_mode,
            clip_resolution=clip_resolution,
            fingerprint_threshold=fingerprint_threshold,
            gdrive_upload=False,
            gdrive_folder_id=gdrive_folder_id,
            gdrive_client_secrets=gdrive_client_secrets,
            gdrive_token=gdrive_token,
            gdrive_include_tmp=gdrive_include_tmp,
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

if run_clicked:
    error = _validate_inputs(source_mode, url_value, file_value)
    if error:
        st.error(error)
    elif gdrive_upload and not gdrive_folder_id.strip():
        st.error("Drive folder ID is required when upload is enabled.")
    else:
        config = _build_config(
            source_mode=source_mode,
            url_value=url_value,
            file_value=file_value,
            outdir_value=outdir_value,
            audio_clips=audio_clips,
            min_segment=min_segment,
            max_segment=max_segment,
            use_acoustid=use_acoustid,
            ref_library=ref_library,
            device=device,
            sample_rate=sample_rate,
            merge_gap=merge_gap,
            exclude_start_seconds=exclude_start_seconds,
            expected_song_count=expected_song_count,
            clip_mode=clip_mode,
            clip_resolution=clip_resolution,
            fingerprint_threshold=fingerprint_threshold,
            gdrive_upload=gdrive_upload,
            gdrive_folder_id=gdrive_folder_id,
            gdrive_client_secrets=gdrive_client_secrets,
            gdrive_token=gdrive_token,
            gdrive_include_tmp=gdrive_include_tmp,
        )

        with st.spinner("Running pipeline..."):
            try:
                result = run_pipeline(config)
                if result == 0:
                    st.success("Pipeline finished.")
                else:
                    st.warning("Pipeline finished with a non-zero code.")
            except Exception as exc:  # pragma: no cover - UI error handling
                st.error(f"Pipeline failed: {exc}")

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
