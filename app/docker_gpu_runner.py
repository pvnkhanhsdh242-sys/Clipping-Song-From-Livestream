"""Run the app pipeline in the shared GPU Docker image from a local UI."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.config import AppConfig


DEFAULT_GPU_IMAGE = "karaoke-clipper:gpu"
TRAIN_GPU_IMAGE = "karaoke-clipper:train-gpu"
CONTAINER_PROJECT_ROOT = "/app"
DOCKER_ENV_KEYS = ("ACOUSTID_API_KEY", "GDRIVE_FOLDER_ID", "PYTHONWARNINGS")


@dataclass(frozen=True)
class DockerMount:
    host_path: Path
    container_path: str

    def as_volume_arg(self) -> str:
        return f"{self.host_path}:{self.container_path}"


@dataclass(frozen=True)
class DockerPipelinePlan:
    command: list[str]
    app_args: list[str]
    mounts: list[DockerMount]


class DockerGpuRunnerError(RuntimeError):
    """Raised when Docker GPU pipeline execution cannot be started."""


class DockerPathMapper:
    """Map local paths into the GPU container while keeping data on the host."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.expanduser().resolve()
        self._external_mounts: dict[Path, str] = {}

    @property
    def mounts(self) -> list[DockerMount]:
        mounts = [DockerMount(self.project_root, CONTAINER_PROJECT_ROOT)]
        mounts.extend(
            DockerMount(host_path=host_path, container_path=container_path)
            for host_path, container_path in self._external_mounts.items()
        )
        return mounts

    def containerize(self, path: Path, *, directory: bool = False, create: bool = False) -> str:
        resolved = path.expanduser().resolve()
        if create:
            target = resolved if directory else resolved.parent
            target.mkdir(parents=True, exist_ok=True)

        relative = _relative_to(resolved, self.project_root)
        if relative is not None:
            return _posix_join(CONTAINER_PROJECT_ROOT, relative)

        if directory:
            mount_root = resolved
            relative_part = Path()
        else:
            mount_root = resolved.parent
            relative_part = Path(resolved.name)

        mount_root = mount_root.resolve()
        container_root = self._external_mounts.get(mount_root)
        if container_root is None:
            container_root = f"/mnt/host{len(self._external_mounts)}"
            self._external_mounts[mount_root] = container_root

        return _posix_join(container_root, relative_part)


def build_docker_pipeline_plan(
    config: AppConfig,
    *,
    project_root: Path | None = None,
    image: str = DEFAULT_GPU_IMAGE,
) -> DockerPipelinePlan:
    root = (project_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    mapper = DockerPathMapper(root)
    app_args = _build_app_args(config, mapper)

    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-w",
        CONTAINER_PROJECT_ROOT,
    ]

    for key in DOCKER_ENV_KEYS:
        value = os.getenv(key)
        if value:
            command.extend(["-e", f"{key}={value}"])

    for mount in mapper.mounts:
        command.extend(["-v", mount.as_volume_arg()])

    command.extend(
        [
            image,
            "python",
            "scripts/container_runtime.py",
            "pipeline",
            "--require-cuda",
            "--",
            *app_args,
        ]
    )
    return DockerPipelinePlan(command=command, app_args=app_args, mounts=mapper.mounts)


def ensure_gpu_image(
    *,
    project_root: Path | None = None,
    image: str = DEFAULT_GPU_IMAGE,
    build_if_missing: bool = True,
    log_callback: Callable[[str], None] | None = None,
) -> None:
    root = (project_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    _ensure_docker_engine(log_callback=log_callback)

    if _run_simple(["docker", "image", "inspect", image], check=False) == 0:
        return

    if not build_if_missing:
        raise DockerGpuRunnerError(
            f"Docker image {image!r} is missing. Build it with scripts\\windows\\docker_train_singing_gpu.bat --build."
        )

    _emit(log_callback, f"Docker image {image} not found. Building shared GPU image...")
    build_command = [
        "docker",
        "build",
        "-f",
        "Dockerfile.gpu",
        "--target",
        "base-gpu",
        "-t",
        DEFAULT_GPU_IMAGE,
        "-t",
        TRAIN_GPU_IMAGE,
        ".",
    ]
    code = _run_streaming(build_command, cwd=root, log_callback=log_callback)
    if code != 0:
        raise DockerGpuRunnerError(f"GPU image build failed with exit code {code}.")


def run_pipeline_in_docker(
    config: AppConfig,
    *,
    project_root: Path | None = None,
    image: str = DEFAULT_GPU_IMAGE,
    build_if_missing: bool = True,
    log_callback: Callable[[str], None] | None = None,
) -> int:
    root = (project_root or Path(__file__).resolve().parents[1]).expanduser().resolve()
    ensure_gpu_image(
        project_root=root,
        image=image,
        build_if_missing=build_if_missing,
        log_callback=log_callback,
    )
    plan = build_docker_pipeline_plan(config, project_root=root, image=image)
    _emit(log_callback, "Starting Docker GPU pipeline...")
    return _run_streaming(plan.command, cwd=root, log_callback=log_callback)


def _build_app_args(config: AppConfig, mapper: DockerPathMapper) -> list[str]:
    args: list[str] = []
    if config.url:
        args.extend(["--url", config.url])
    elif config.file:
        args.extend(["--file", mapper.containerize(config.file)])
    else:
        raise DockerGpuRunnerError("Pipeline config must include either url or file.")

    args.extend(
        [
            "--profile",
            "custom",
            "--outdir",
            mapper.containerize(config.outdir, directory=True, create=True),
            "--audio-clips",
            _bool_arg(config.audio_clips),
            "--min-segment",
            str(config.min_segment),
            "--max-segment",
            str(config.max_segment),
            "--merge-max-segment",
            str(config.merge_max_segment),
            "--segment-tolerance",
            str(config.segment_tolerance),
            "--pre-roll",
            str(config.pre_roll_sec),
            "--post-roll",
            str(config.post_roll_sec),
            "--bridge-noise-gap",
            str(config.bridge_noise_gap_sec),
            "--bridge-speech-gap",
            str(config.bridge_speech_gap_sec),
            "--use-acoustid",
            _bool_arg(config.use_acoustid),
            "--sample-rate",
            str(config.sample_rate),
            "--merge-gap",
            str(config.merge_gap),
            "--clip-mode",
            config.clip_mode,
            "--clip-resolution",
            config.clip_resolution,
            "--fingerprint-threshold",
            str(config.fingerprint_threshold),
            "--whisperx-boundary-mode",
            config.whisperx_boundary_mode,
            "--whisperx-max-start-shrink",
            str(config.whisperx_max_start_shrink_sec),
            "--whisperx-max-end-shrink",
            str(config.whisperx_max_end_shrink_sec),
            "--allow-hard-split",
            _bool_arg(config.allow_hard_split),
            "--energy-frame-ms",
            str(config.energy_frame_ms),
            "--energy-min-active-ms",
            str(config.energy_min_active_ms),
            "--energy-min-silence-ms",
            str(config.energy_min_silence_ms),
            "--review-score-threshold",
            str(config.review_score_threshold),
            "--music-ratio-threshold",
            str(config.music_ratio_threshold),
            "--singing-score-threshold",
            str(config.singing_score_threshold),
            "--singing-model-mode",
            config.singing_model_mode,
            "--gdrive-upload",
            _bool_arg(config.gdrive_upload),
            "--gdrive-token",
            mapper.containerize(config.gdrive_token_path),
            "--gdrive-include-tmp",
            _bool_arg(config.gdrive_include_tmp),
            "--gdrive-upload-mode",
            config.gdrive_upload_mode,
            "--exclude-start-seconds",
            str(config.exclude_start_seconds),
            "--exclude-end-seconds",
            str(config.exclude_end_seconds),
        ]
    )

    if config.ref_library is not None:
        args.extend(["--ref-library", mapper.containerize(config.ref_library)])
    if config.expected_song_count is not None:
        args.extend(["--expected-song-count", str(config.expected_song_count)])
    if config.singing_model_path is not None:
        is_dir = config.singing_model_path.exists() and config.singing_model_path.is_dir()
        args.extend(["--singing-model-path", mapper.containerize(config.singing_model_path, directory=is_dir)])
    if config.gdrive_folder_id:
        args.extend(["--gdrive-folder-id", config.gdrive_folder_id])
    if config.gdrive_client_secrets is not None:
        is_dir = config.gdrive_client_secrets.exists() and config.gdrive_client_secrets.is_dir()
        args.extend(["--gdrive-client-secrets", mapper.containerize(config.gdrive_client_secrets, directory=is_dir)])

    return args


def _run_simple(command: list[str], message: str | None = None, *, check: bool = True) -> int:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as exc:
        if check:
            raise DockerGpuRunnerError(message or f"Command not found: {command[0]}") from exc
        return 127

    if check and completed.returncode != 0:
        raise DockerGpuRunnerError(message or f"Command failed: {' '.join(command)}")
    return int(completed.returncode)


def _ensure_docker_engine(log_callback: Callable[[str], None] | None = None) -> None:
    if _run_simple(["docker", "info"], check=False) == 0:
        return

    docker_desktop = _find_docker_desktop()
    if docker_desktop is not None:
        _emit(log_callback, "Docker engine is not ready. Starting Docker Desktop...")
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        subprocess.Popen(
            [str(docker_desktop)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
        for _ in range(120):
            if _run_simple(["docker", "info"], check=False) == 0:
                return
            time.sleep(1)

    raise DockerGpuRunnerError(
        "Docker Desktop/Engine is not running. Start Docker Desktop, then run the pipeline again."
    )


def _find_docker_desktop() -> Path | None:
    if os.name != "nt":
        return None
    candidates = [
        Path(os.environ.get("ProgramFiles", "")) / "Docker" / "Docker" / "Docker Desktop.exe",
        Path(os.environ.get("LocalAppData", "")) / "Programs" / "Docker" / "Docker" / "Docker Desktop.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _run_streaming(
    command: list[str],
    *,
    cwd: Path,
    log_callback: Callable[[str], None] | None = None,
) -> int:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=creationflags,
    )
    assert process.stdout is not None
    for line in process.stdout:
        _emit(log_callback, line.rstrip())
    return int(process.wait())


def _emit(log_callback: Callable[[str], None] | None, message: str) -> None:
    if log_callback:
        log_callback(message)


def _bool_arg(value: bool) -> str:
    return "true" if value else "false"


def _relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _posix_join(root: str, relative: Path) -> str:
    rel = relative.as_posix()
    if not rel or rel == ".":
        return root
    return f"{root.rstrip('/')}/{rel}"
