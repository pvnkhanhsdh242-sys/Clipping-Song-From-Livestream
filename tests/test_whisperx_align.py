import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.align.whisperx_align import WhisperXRefiner


def test_refine_segment_disables_whisperx_vad(tmp_path: Path):
    logger = logging.getLogger("test")
    probe_audio = tmp_path / "probe.wav"
    probe_audio.write_bytes(b"fake")
    payload = {"waveform": object(), "sample_rate": 16000}

    dummy_model = MagicMock()
    dummy_model.transcribe.return_value = {
        "segments": [
            {"start": 0.2, "end": 1.4},
        ]
    }

    fake_whisperx = SimpleNamespace(load_model=MagicMock(return_value=dummy_model))

    with patch.dict("sys.modules", {"whisperx": fake_whisperx}):
        with patch("app.align.whisperx_align.extract_temp_wav_segment", return_value=probe_audio):
            with patch("app.align.whisperx_align._build_audio_payload", return_value=payload):
                refiner = WhisperXRefiner(device="cpu", logger=logger)
                start_sec, end_sec = refiner.refine_segment(Path("input.wav"), 10.0, 12.0, tmp_path)

    assert dummy_model.transcribe.call_count == 1
    args, kwargs = dummy_model.transcribe.call_args
    assert args[0] == payload
    assert kwargs.get("vad_filter") is False
    assert start_sec == 9.2
    assert end_sec == 12.0


def test_refine_segment_skips_when_vad_filter_unsupported(tmp_path: Path):
    logger = logging.getLogger("test")
    probe_audio = tmp_path / "probe.wav"
    probe_audio.write_bytes(b"fake")
    payload = {"waveform": object(), "sample_rate": 16000}

    class DummyModel:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def transcribe(self, audio_input, **kwargs):
            self.calls.append({"audio_input": audio_input, **kwargs})
            raise TypeError("FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'vad_filter'")

    dummy_model = DummyModel()
    fake_whisperx = SimpleNamespace(load_model=MagicMock(return_value=dummy_model))

    with patch.dict("sys.modules", {"whisperx": fake_whisperx}):
        with patch("app.align.whisperx_align.extract_temp_wav_segment", return_value=probe_audio):
            with patch("app.align.whisperx_align._build_audio_payload", return_value=payload):
                refiner = WhisperXRefiner(device="cpu", logger=logger)
                start_sec, end_sec = refiner.refine_segment(Path("input.wav"), 10.0, 12.0, tmp_path)

    assert len(dummy_model.calls) == 1
    assert dummy_model.calls[0].get("audio_input") == payload
    assert dummy_model.calls[0].get("vad_filter") is False
    assert start_sec == 10.0
    assert end_sec == 12.0
