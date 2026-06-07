from pathlib import Path

from app.config import AppConfig
from app.docker_gpu_runner import build_docker_pipeline_plan


def _config(project_root: Path, source_file: Path | None = None, url: str | None = None) -> AppConfig:
    ref_library = project_root / "data" / "reference_library.json"
    singing_model = project_root / "data" / "models" / "singing_candidate"
    token = project_root / "secret" / "token.json"
    client_secrets = project_root / "secret"
    ref_library.parent.mkdir(parents=True, exist_ok=True)
    singing_model.mkdir(parents=True, exist_ok=True)
    token.parent.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        url=url,
        file=source_file,
        outdir=project_root / "output",
        audio_clips=True,
        min_segment=30.0,
        max_segment=420.0,
        use_acoustid=False,
        ref_library=ref_library,
        device="cpu",
        sample_rate=16000,
        merge_gap=1.5,
        merge_max_segment=420.0,
        segment_tolerance=0.0,
        pre_roll_sec=0.5,
        post_roll_sec=2.0,
        bridge_noise_gap_sec=2.0,
        bridge_speech_gap_sec=1.0,
        expected_song_count=12,
        clip_mode="accurate",
        clip_resolution="source",
        fingerprint_threshold=0.45,
        acoustid_api_key=None,
        whisperx_boundary_mode="safe",
        whisperx_max_start_shrink_sec=0.5,
        whisperx_max_end_shrink_sec=0.5,
        allow_hard_split=False,
        energy_frame_ms=100,
        energy_min_active_ms=500,
        energy_min_silence_ms=1200,
        profile="karaoke",
        review_score_threshold=0.65,
        music_ratio_threshold=0.0,
        singing_model_path=singing_model,
        singing_score_threshold=0.5,
        singing_model_mode="score",
        gdrive_upload=False,
        gdrive_folder_id=None,
        gdrive_client_secrets=client_secrets,
        gdrive_token_path=token,
        gdrive_include_tmp=False,
        gdrive_upload_mode="clips",
        exclude_start_seconds=0.0,
        exclude_end_seconds=0.0,
    )


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_docker_pipeline_plan_mounts_project_and_external_source(tmp_path: Path):
    project_root = tmp_path / "repo"
    source_dir = tmp_path / "incoming"
    project_root.mkdir()
    source_dir.mkdir()
    source_file = source_dir / "vod.mp4"
    source_file.write_bytes(b"placeholder")

    plan = build_docker_pipeline_plan(_config(project_root, source_file=source_file), project_root=project_root)

    assert plan.command[:5] == ["docker", "run", "--rm", "--gpus", "all"]
    assert "karaoke-clipper:gpu" in plan.command
    image_index = plan.command.index("karaoke-clipper:gpu")
    assert plan.command[image_index + 1 : image_index + 6] == [
        "python",
        "scripts/container_runtime.py",
        "pipeline",
        "--require-cuda",
        "--",
    ]
    assert plan.mounts[0].host_path == project_root.resolve()
    assert plan.mounts[0].container_path == "/app"
    assert plan.mounts[1].host_path == source_dir.resolve()
    assert plan.mounts[1].container_path == "/mnt/host0"
    assert _arg_value(plan.app_args, "--file") == "/mnt/host0/vod.mp4"
    assert _arg_value(plan.app_args, "--outdir") == "/app/output"
    assert _arg_value(plan.app_args, "--ref-library") == "/app/data/reference_library.json"
    assert _arg_value(plan.app_args, "--singing-model-path") == "/app/data/models/singing_candidate"
    assert _arg_value(plan.app_args, "--gdrive-token") == "/app/secret/token.json"
    assert _arg_value(plan.app_args, "--profile") == "custom"
    assert _arg_value(plan.app_args, "--expected-song-count") == "12"


def test_docker_pipeline_plan_uses_url_without_external_source_mount(tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()

    plan = build_docker_pipeline_plan(
        _config(project_root, url="https://www.youtube.com/watch?v=abc123"),
        project_root=project_root,
    )

    assert _arg_value(plan.app_args, "--url") == "https://www.youtube.com/watch?v=abc123"
    assert "--file" not in plan.app_args
    assert [mount.container_path for mount in plan.mounts] == ["/app"]
