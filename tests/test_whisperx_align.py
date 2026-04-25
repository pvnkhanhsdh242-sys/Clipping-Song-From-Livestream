import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.align.whisperx_align import WhisperXRefiner


def test_refine_segment_disables_whisperx_vad(tmp_path: Path):
    logger = logging.getLogger("test")
    probe_audio = tmp_path / "probe.wav"
    probe_audio.write_bytes(b"fake")

    dummy_model = MagicMock()
    dummy_model.transcribe.return_value = {
        "segments": [
            {"start": 0.2, "end": 1.4},
        ]
    }

    fake_whisperx = SimpleNamespace(load_model=MagicMock(return_value=dummy_model))

    with patch.dict("sys.modules", {"whisperx": fake_whisperx}):
        with patch("app.align.whisperx_align.extract_temp_wav_segment", return_value=probe_audio):
            refiner = WhisperXRefiner(device="cpu", logger=logger)
            start_sec, end_sec = refiner.refine_segment(Path("input.wav"), 10.0, 12.0, tmp_path)

    assert dummy_model.transcribe.call_count == 1
    _, kwargs = dummy_model.transcribe.call_args
    assert kwargs.get("vad_filter") is False
    assert start_sec == 9.2
    assert end_sec == 10.4


def test_refine_segment_retries_without_vad_filter_when_unsupported(tmp_path: Path):
    logger = logging.getLogger("test")
    probe_audio = tmp_path / "probe.wav"
    probe_audio.write_bytes(b"fake")

    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe(self, audio_path: str, **kwargs):
            self.calls.append({"audio_path": audio_path, **kwargs})
            if "vad_filter" in kwargs:
                raise TypeError("FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'vad_filter'")
            return {"segments": [{"start": 0.1, "end": 1.1}]}

    dummy_model = DummyModel()
    fake_whisperx = SimpleNamespace(load_model=MagicMock(return_value=dummy_model))

    with patch.dict("sys.modules", {"whisperx": fake_whisperx}):
        with patch("app.align.whisperx_align.extract_temp_wav_segment", return_value=probe_audio):
            refiner = WhisperXRefiner(device="cpu", logger=logger)
            start_sec, end_sec = refiner.refine_segment(Path("input.wav"), 10.0, 12.0, tmp_path)

    assert len(dummy_model.calls) == 2
    assert dummy_model.calls[0].get("vad_filter") is False
    assert "vad_filter" not in dummy_model.calls[1]
    assert start_sec == 9.1
    assert end_sec == 10.1
