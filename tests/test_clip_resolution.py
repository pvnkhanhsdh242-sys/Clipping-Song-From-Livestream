import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.clip.cutter import _accurate_clip_command, export_clip


def test_accurate_command_uses_source_resolution_by_default():
    command = _accurate_clip_command(Path("input.mp4"), 1.0, 2.0, Path("out.mp4"), "source")

    assert "-vf" not in command


def test_accurate_command_applies_720p_filter():
    command = _accurate_clip_command(Path("input.mp4"), 1.0, 2.0, Path("out.mp4"), "720p")

    assert "-vf" in command
    filter_value = command[command.index("-vf") + 1]
    assert "scale=w=1280:h=720" in filter_value
    assert "pad=1280:720" in filter_value


def test_accurate_command_uses_nvenc_for_cuda_device():
    command = _accurate_clip_command(Path("input.mp4"), 1.0, 2.0, Path("out.mp4"), "source", device="cuda")

    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert "-cq" in command


def test_accurate_command_uses_x264_for_cpu_device():
    command = _accurate_clip_command(Path("input.mp4"), 1.0, 2.0, Path("out.mp4"), "source", device="cpu")

    assert command[command.index("-c:v") + 1] == "libx264"
    assert "-crf" in command


def test_fast_mode_with_fixed_resolution_switches_to_accurate(tmp_path: Path):
    logger = logging.getLogger("test")

    with patch("app.clip.cutter.run_command") as run_command_mock:
        run_command_mock.return_value = MagicMock(returncode=0)

        export_clip(
            video_path=Path("input.mp4"),
            start_sec=0.0,
            end_sec=3.0,
            clips_dir=tmp_path,
            clip_stem="clip_001",
            include_audio_clip=False,
            mode="fast",
            clip_resolution="1080p",
            device="cpu",
            logger=logger,
        )

    assert run_command_mock.call_count == 1
    first_command = run_command_mock.call_args_list[0].args[0]
    assert "-vf" in first_command


def test_fast_mode_with_fixed_resolution_uses_nvenc_when_cuda(tmp_path: Path):
    logger = logging.getLogger("test")

    with patch("app.clip.cutter.run_command") as run_command_mock:
        run_command_mock.return_value = MagicMock(returncode=0)

        export_clip(
            video_path=Path("input.mp4"),
            start_sec=0.0,
            end_sec=3.0,
            clips_dir=tmp_path,
            clip_stem="clip_001",
            include_audio_clip=False,
            mode="fast",
            clip_resolution="1080p",
            device="cuda",
            logger=logger,
        )

    first_command = run_command_mock.call_args_list[0].args[0]
    assert first_command[first_command.index("-c:v") + 1] == "h264_nvenc"


def test_accurate_mode_retries_cpu_when_nvenc_fails(tmp_path: Path):
    logger = logging.getLogger("test")

    with patch("app.clip.cutter.run_command") as run_command_mock:
        run_command_mock.side_effect = [MagicMock(returncode=1), MagicMock(returncode=0)]

        export_clip(
            video_path=Path("input.mp4"),
            start_sec=0.0,
            end_sec=3.0,
            clips_dir=tmp_path,
            clip_stem="clip_001",
            include_audio_clip=False,
            mode="accurate",
            clip_resolution="source",
            device="cuda",
            logger=logger,
        )

    assert run_command_mock.call_count == 2
    nvenc_command = run_command_mock.call_args_list[0].args[0]
    cpu_command = run_command_mock.call_args_list[1].args[0]
    assert nvenc_command[nvenc_command.index("-c:v") + 1] == "h264_nvenc"
    assert cpu_command[cpu_command.index("-c:v") + 1] == "libx264"


def test_fast_mode_source_keeps_stream_copy(tmp_path: Path):
    logger = logging.getLogger("test")

    with patch("app.clip.cutter.run_command") as run_command_mock:
        run_command_mock.return_value = MagicMock(returncode=0)

        export_clip(
            video_path=Path("input.mp4"),
            start_sec=0.0,
            end_sec=3.0,
            clips_dir=tmp_path,
            clip_stem="clip_001",
            include_audio_clip=False,
            mode="fast",
            clip_resolution="source",
            device="cpu",
            logger=logger,
        )

    assert run_command_mock.call_count == 1
    first_command = run_command_mock.call_args_list[0].args[0]
    assert first_command[first_command.index("-c") + 1] == "copy"


def test_fast_mode_source_does_not_reencode_even_when_cuda(tmp_path: Path):
    logger = logging.getLogger("test")

    with patch("app.clip.cutter.run_command") as run_command_mock:
        run_command_mock.return_value = MagicMock(returncode=0)

        export_clip(
            video_path=Path("input.mp4"),
            start_sec=0.0,
            end_sec=3.0,
            clips_dir=tmp_path,
            clip_stem="clip_001",
            include_audio_clip=False,
            mode="fast",
            clip_resolution="source",
            device="cuda",
            logger=logger,
        )

    first_command = run_command_mock.call_args_list[0].args[0]
    assert first_command[first_command.index("-c") + 1] == "copy"
    assert "h264_nvenc" not in first_command
