"""Optional PyTorch backend for singing candidate scoring."""

from __future__ import annotations

import json
import logging
import math
import os
import random
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

from app.singing.labels import LabeledCandidate, load_labeled_candidates


MODEL_TYPE = "pytorch_logspec_cnn"
MODEL_FILE_NAME = "model.pt"
METADATA_FILE_NAME = "metadata.json"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FREQ_BINS = 64
DEFAULT_TIME_FRAMES = 128


@dataclass(frozen=True)
class PyTorchArtifact:
    model_path: Path
    metadata_path: Path
    metadata: dict[str, object]


def _import_numpy():
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyTorch singing training requires numpy.") from exc
    return np


def _import_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("PyTorch singing backend requires torch.") from exc
    return torch


def _import_torch_nn():
    torch = _import_torch()
    import torch.nn as nn  # type: ignore

    return torch, nn


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve auto/cpu/cuda against the current PyTorch runtime."""
    requested = str(requested or "auto").lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of auto, cpu, or cuda")

    forced = os.environ.get("KARAOKE_FORCE_DEVICE")
    if requested == "auto" and forced in {"cpu", "cuda"}:
        requested = forced

    if requested == "cpu":
        return "cpu"

    torch = _import_torch()
    cuda_available = bool(torch.cuda.is_available())
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested for PyTorch singing training, but torch.cuda is not available.")
        return "cuda"

    return "cuda" if cuda_available else "cpu"


class LogSpecCNN(_import_torch_nn()[1].Module):  # type: ignore[misc]
    """Small CNN over normalized log-spectrogram windows."""

    def __init__(self, freq_bins: int = DEFAULT_FREQ_BINS, time_frames: int = DEFAULT_TIME_FRAMES) -> None:
        torch, nn = _import_torch_nn()
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.time_frames = int(time_frames)
        self.net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(32, 1),
        )

    def forward(self, inputs):  # noqa: D401 - PyTorch conventional method
        return self.net(inputs).squeeze(1)


def _read_wav_window(audio_path: Path, start_sec: float, end_sec: float):
    np = _import_numpy()
    with wave.open(str(audio_path), "rb") as wav_handle:
        sample_rate = int(wav_handle.getframerate())
        channels = int(wav_handle.getnchannels())
        sample_width = int(wav_handle.getsampwidth())
        frame_count = int(wav_handle.getnframes())
        if sample_width != 2:
            raise RuntimeError("PyTorch singing backend expects 16-bit PCM WAV input.")

        start_frame = max(0, min(frame_count, int(float(start_sec) * sample_rate)))
        end_frame = max(start_frame, min(frame_count, int(float(end_sec) * sample_rate)))
        wav_handle.setpos(start_frame)
        payload = wav_handle.readframes(end_frame - start_frame)

    samples = np.frombuffer(payload, dtype=np.int16)
    if samples.size == 0:
        return np.array([], dtype=np.float32), sample_rate
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples.astype(np.float32) / 32768.0, sample_rate


def _extract_media_window(media_path: Path, start_sec: float, end_sec: float, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-to",
        f"{end_sec:.3f}",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(DEFAULT_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting {media_path}: {completed.stderr.strip()}")
    return output_wav


def _candidate_audio_window(candidate: LabeledCandidate, tmp_dir: Path, index: int) -> tuple[Path, float, float]:
    source_path = candidate.source_video
    if source_path.suffix.lower() == ".wav":
        return source_path, candidate.start_sec, candidate.end_sec

    output_wav = tmp_dir / f"candidate_{index:06d}.wav"
    _extract_media_window(source_path, candidate.start_sec, candidate.end_sec, output_wav)
    duration = max(0.0, candidate.end_sec - candidate.start_sec)
    return output_wav, 0.0, duration


def _window_offsets(total_samples: int, window_samples: int, count: int) -> list[int]:
    count = max(1, int(count))
    if total_samples <= 0 or window_samples <= 0 or total_samples <= window_samples:
        return [0 for _ in range(count)]
    if count == 1:
        return [max(0, (total_samples - window_samples) // 2)]
    max_offset = total_samples - window_samples
    return [int(round(value)) for value in _import_numpy().linspace(0, max_offset, num=count)]


def _normalize_window(samples):
    np = _import_numpy()
    if samples.size == 0:
        return samples
    mean = float(np.mean(samples))
    std = float(np.std(samples))
    if std <= 1e-6:
        return samples - mean
    return (samples - mean) / std


def _logspec_tensor(samples, sample_rate: int, *, freq_bins: int, time_frames: int):
    torch = _import_torch()
    import torch.nn.functional as F  # type: ignore

    waveform = torch.as_tensor(samples, dtype=torch.float32)
    if waveform.numel() == 0:
        waveform = torch.zeros(max(1, int(sample_rate)), dtype=torch.float32)

    n_fft = min(512, max(64, int(waveform.numel())))
    win_length = min(400, n_fft)
    hop_length = max(1, win_length // 2)
    window = torch.hann_window(win_length)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    logspec = torch.log1p(spec)
    logspec = logspec.unsqueeze(0).unsqueeze(0)
    logspec = F.interpolate(logspec, size=(int(freq_bins), int(time_frames)), mode="bilinear", align_corners=False)
    logspec = logspec.squeeze(0)
    mean = logspec.mean()
    std = logspec.std()
    if float(std) > 1e-6:
        logspec = (logspec - mean) / std
    else:
        logspec = logspec - mean
    return logspec


def logspec_windows_for_wav(
    audio_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    window_sec: float,
    windows_per_clip: int,
    freq_bins: int = DEFAULT_FREQ_BINS,
    time_frames: int = DEFAULT_TIME_FRAMES,
):
    """Return fixed-size log-spectrogram tensors sampled across one WAV segment."""
    np = _import_numpy()
    torch = _import_torch()
    samples, sample_rate = _read_wav_window(audio_path, start_sec, end_sec)
    window_samples = max(1, int(round(float(window_sec) * sample_rate)))
    offsets = _window_offsets(int(samples.size), window_samples, windows_per_clip)
    tensors = []
    for offset in offsets:
        chunk = samples[offset : offset + window_samples]
        if chunk.size < window_samples:
            chunk = np.pad(chunk, (0, window_samples - chunk.size), mode="constant")
        chunk = _normalize_window(chunk.astype(np.float32, copy=False))
        tensors.append(_logspec_tensor(chunk, sample_rate, freq_bins=freq_bins, time_frames=time_frames))
    return torch.stack(tensors, dim=0)


def _build_window_dataset(
    candidates: Sequence[LabeledCandidate],
    *,
    tmp_dir: Path,
    window_sec: float,
    windows_per_clip: int,
    freq_bins: int,
    time_frames: int,
) -> tuple[object, object]:
    torch = _import_torch()
    rows = []
    labels = []
    for index, candidate in enumerate(candidates, start=1):
        audio_path, start_sec, end_sec = _candidate_audio_window(candidate, tmp_dir, index)
        windows = logspec_windows_for_wav(
            audio_path,
            start_sec,
            end_sec,
            window_sec=window_sec,
            windows_per_clip=windows_per_clip,
            freq_bins=freq_bins,
            time_frames=time_frames,
        )
        rows.append(windows)
        labels.extend([float(candidate.label_singing)] * int(windows.shape[0]))
    return torch.cat(rows, dim=0), torch.as_tensor(labels, dtype=torch.float32)


def _validation_indices(labels, validation_size: float, random_state: int) -> tuple[list[int], list[int]]:
    np = _import_numpy()
    y = labels.detach().cpu().numpy().astype(int)
    total = int(y.size)
    validation_count = int(math.ceil(float(total) * float(validation_size)))
    unique, counts = np.unique(y, return_counts=True)
    can_validate = (
        total >= 4
        and len(unique) == 2
        and min(int(count) for count in counts) >= 2
        and 2 <= validation_count <= total - 2
    )
    indices = list(range(total))
    rng = random.Random(int(random_state))
    rng.shuffle(indices)
    if not can_validate:
        return indices, []

    val_indices: list[int] = []
    train_indices: list[int] = []
    by_label: dict[int, list[int]] = {0: [], 1: []}
    for index in indices:
        by_label[int(y[index])].append(index)
    for label_indices in by_label.values():
        label_val_count = max(1, int(round(len(label_indices) * float(validation_size))))
        val_indices.extend(label_indices[:label_val_count])
        train_indices.extend(label_indices[label_val_count:])
    if not train_indices or not val_indices:
        return indices, []
    return train_indices, val_indices


def _make_loader(features, labels, indices: list[int], *, batch_size: int, shuffle: bool, seed: int):
    torch = _import_torch()
    from torch.utils.data import DataLoader, TensorDataset  # type: ignore

    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(features.index_select(0, index_tensor), labels.index_select(0, index_tensor))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=shuffle, generator=generator)


def _evaluate_model(model, features, labels, indices: list[int], device: str) -> dict[str, float | int | None]:
    torch = _import_torch()
    if not indices:
        return {"validation_count": 0, "validation_accuracy": None, "validation_roc_auc": None}
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    x_val = features.index_select(0, index_tensor).to(device)
    y_val = labels.index_select(0, index_tensor).to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_val)).detach().cpu()
    predictions = (probabilities >= 0.5).float()
    y_cpu = y_val.detach().cpu()
    accuracy = float((predictions == y_cpu).float().mean().item())
    metrics: dict[str, float | int | None] = {
        "validation_count": int(len(indices)),
        "validation_accuracy": accuracy,
        "validation_roc_auc": None,
    }
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore

        if len(set(float(value) for value in y_cpu.tolist())) == 2:
            metrics["validation_roc_auc"] = float(roc_auc_score(y_cpu.numpy(), probabilities.numpy()))
    except Exception:
        metrics["validation_roc_auc"] = None
    return metrics


def _positive_weight(labels, indices: list[int]):
    torch = _import_torch()
    selected = labels[torch.as_tensor(indices, dtype=torch.long)]
    positives = float((selected == 1).sum().item())
    negatives = float((selected == 0).sum().item())
    if positives <= 0:
        return torch.as_tensor(1.0, dtype=torch.float32)
    return torch.as_tensor(max(1.0, negatives / positives), dtype=torch.float32)


def train_pytorch_singing_candidate_model(
    manifest_paths: Sequence[Path],
    output_dir: Path,
    *,
    validation_size: float = 0.25,
    random_state: int = 13,
    epochs: int = 5,
    device: str = "auto",
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    window_sec: float = 12.0,
    windows_per_clip: int = 4,
    logger: logging.Logger | None = None,
    result_factory: Callable[..., object] | None = None,
) -> object:
    """Train and persist the optional PyTorch log-spectrogram CNN backend."""
    torch = _import_torch()
    import torch.nn as nn  # type: ignore

    if epochs <= 0:
        raise ValueError("epochs must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if window_sec <= 0:
        raise ValueError("window_sec must be > 0")
    if windows_per_clip <= 0:
        raise ValueError("windows_per_clip must be > 0")

    log = logger or logging.getLogger("karaoke_clipper.singing_training")
    candidates = load_labeled_candidates(manifest_paths)
    if not candidates:
        raise ValueError("No labeled rows found. Add label_singing to reviewed manifests first.")

    torch.manual_seed(int(random_state))
    random.seed(int(random_state))

    with tempfile.TemporaryDirectory(prefix="singing_pytorch_training_") as tmp:
        features, labels = _build_window_dataset(
            candidates,
            tmp_dir=Path(tmp),
            window_sec=window_sec,
            windows_per_clip=windows_per_clip,
            freq_bins=DEFAULT_FREQ_BINS,
            time_frames=DEFAULT_TIME_FRAMES,
        )

    resolved_device = resolve_torch_device(device)
    positive_count = int(sum(1 for candidate in candidates if candidate.label_singing == 1))
    negative_count = int(sum(1 for candidate in candidates if candidate.label_singing == 0))
    train_indices, val_indices = _validation_indices(labels, validation_size, random_state)
    if not val_indices:
        log.warning("Too few PyTorch windows for validation split; training on all windows.")

    model = LogSpecCNN(DEFAULT_FREQ_BINS, DEFAULT_TIME_FRAMES).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.BCEWithLogitsLoss(pos_weight=_positive_weight(labels, train_indices).to(resolved_device))
    loader = _make_loader(features, labels, train_indices, batch_size=batch_size, shuffle=True, seed=random_state)

    last_loss = None
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device)
            batch_y = batch_y.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item())
            batch_count += 1
        last_loss = total_loss / max(1, batch_count)
        log.info("PyTorch singing epoch %s/%s loss=%.6f", epoch, epochs, last_loss)

    metrics: dict[str, float | int | str | None] = {
        "labeled_count": int(len(candidates)),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "window_count": int(labels.numel()),
        "training_window_count": int(len(train_indices)),
        "training_loss": float(last_loss) if last_loss is not None else None,
    }
    metrics.update(_evaluate_model(model, features, labels, val_indices, resolved_device))

    output = output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / MODEL_FILE_NAME
    metadata_path = output / METADATA_FILE_NAME
    model_config = {
        "freq_bins": DEFAULT_FREQ_BINS,
        "time_frames": DEFAULT_TIME_FRAMES,
        "window_sec": float(window_sec),
        "windows_per_clip": int(windows_per_clip),
        "sample_rate": DEFAULT_SAMPLE_RATE,
    }
    torch.save(
        {
            "model_type": MODEL_TYPE,
            "model_config": model_config,
            "state_dict": model.cpu().state_dict(),
        },
        model_path,
    )
    metadata = {
        "schema_version": 1,
        "backend": "pytorch",
        "model_type": MODEL_TYPE,
        "model_file": MODEL_FILE_NAME,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_column": "label_singing",
        "threshold_default": 0.5,
        "training_epochs": int(epochs),
        "device": resolved_device,
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "model_config": model_config,
        "metrics": metrics,
        "manifest_paths": [str(path.expanduser().resolve()) for path in manifest_paths],
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    if result_factory is None:
        return PyTorchArtifact(model_path=model_path, metadata_path=metadata_path, metadata=metadata)
    return result_factory(
        output_dir=output,
        model_path=model_path,
        metadata_path=metadata_path,
        labeled_count=int(len(candidates)),
        positive_count=positive_count,
        negative_count=negative_count,
        metrics=metrics,
    )


class PyTorchCandidateModel:
    """Runtime wrapper for a saved PyTorch singing model artifact."""

    def __init__(
        self,
        *,
        model,
        model_name: str,
        model_config: dict[str, object],
        device: str,
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.model_config = model_config
        self.device = device

    @classmethod
    def load(cls, model_path: Path, metadata: dict[str, object] | None = None, *, device: str = "auto"):
        torch = _import_torch()
        metadata = metadata or {}
        resolved_device = resolve_torch_device(device)
        try:
            payload = torch.load(model_path, map_location=resolved_device, weights_only=False)
        except TypeError:  # pragma: no cover - older torch compatibility
            payload = torch.load(model_path, map_location=resolved_device)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid PyTorch singing model artifact: {model_path}")

        config = payload.get("model_config") or metadata.get("model_config") or {}
        if not isinstance(config, dict):
            config = {}
        freq_bins = int(config.get("freq_bins") or DEFAULT_FREQ_BINS)
        time_frames = int(config.get("time_frames") or DEFAULT_TIME_FRAMES)
        model = LogSpecCNN(freq_bins=freq_bins, time_frames=time_frames)
        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise RuntimeError(f"PyTorch singing model artifact is missing state_dict: {model_path}")
        model.load_state_dict(state_dict)
        model.to(resolved_device)
        model.eval()
        model_name = str(metadata.get("model_type") or payload.get("model_type") or MODEL_TYPE)
        return cls(model=model, model_name=model_name, model_config=config, device=resolved_device)

    def score_candidate(self, audio_path: Path, start_sec: float, end_sec: float) -> float:
        torch = _import_torch()
        window_sec = float(self.model_config.get("window_sec") or 12.0)
        windows_per_clip = int(self.model_config.get("windows_per_clip") or 4)
        freq_bins = int(self.model_config.get("freq_bins") or DEFAULT_FREQ_BINS)
        time_frames = int(self.model_config.get("time_frames") or DEFAULT_TIME_FRAMES)
        windows = logspec_windows_for_wav(
            audio_path,
            start_sec,
            end_sec,
            window_sec=window_sec,
            windows_per_clip=windows_per_clip,
            freq_bins=freq_bins,
            time_frames=time_frames,
        ).to(self.device)
        with torch.no_grad():
            probabilities = torch.sigmoid(self.model(windows)).detach().cpu()
        if probabilities.numel() == 0:
            return 0.0
        score = float(probabilities.mean().item())
        if math.isnan(score):
            return 0.0
        return max(0.0, min(1.0, score))


def load_metadata(metadata_path: Path) -> dict[str, object]:
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    return loaded if isinstance(loaded, dict) else {}
