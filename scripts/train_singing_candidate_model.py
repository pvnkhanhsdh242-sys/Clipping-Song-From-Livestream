"""Train the candidate-level singing scorer from reviewed manifests."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.singing.training import train_singing_candidate_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a singing candidate scorer from reviewed manifests.")
    parser.add_argument(
        "--manifest",
        nargs="+",
        required=True,
        help="One or more reviewed manifest CSV/JSON files with label_singing values.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/models/singing_candidate",
        help="Directory for model artifact and metadata.json.",
    )
    parser.add_argument(
        "--backend",
        choices=["sklearn", "pytorch"],
        default="sklearn",
        help="Training backend. sklearn is the CPU default; pytorch enables the log-spectrogram CNN.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="PyTorch device. Ignored by the sklearn backend.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="PyTorch mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="PyTorch optimizer learning rate.")
    parser.add_argument("--window-sec", type=float, default=12.0, help="PyTorch log-spectrogram window length.")
    parser.add_argument("--windows-per-clip", type=int, default=4, help="PyTorch windows sampled per labeled clip.")
    parser.add_argument("--validation-size", type=float, default=0.25, help="Validation split ratio when enough labels exist.")
    parser.add_argument("--random-state", type=int, default=13, help="Random seed for split/model training.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Training iterations for sklearn, or neural-network epochs for PyTorch.",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be > 0")
    if args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.window_sec <= 0:
        parser.error("--window-sec must be > 0")
    if args.windows_per_clip <= 0:
        parser.error("--windows-per-clip must be > 0")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = train_singing_candidate_model(
        manifest_paths=[Path(value) for value in args.manifest],
        output_dir=Path(args.output_dir),
        backend=args.backend,
        validation_size=args.validation_size,
        random_state=args.random_state,
        max_iter=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        window_sec=args.window_sec,
        windows_per_clip=args.windows_per_clip,
        logger=logging.getLogger("karaoke_clipper.singing_training"),
    )

    print(f"Wrote model: {result.model_path}")
    print(f"Wrote metadata: {result.metadata_path}")
    print(
        "Labels: "
        f"total={result.labeled_count} positive={result.positive_count} negative={result.negative_count}"
    )
    print(f"Metrics: {result.metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
