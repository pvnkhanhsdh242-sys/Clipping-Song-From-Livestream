"""Container runtime helpers for GPU healthchecks and device forcing."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from typing import Any, Sequence

FORCED_DEVICE_ENV = "KARAOKE_FORCE_DEVICE"
CUDA_VISIBLE_ENV = "KARAOKE_CUDA_VISIBLE"


def detect_runtime_device() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "torch_available": False,
        "torch_version": None,
        "cuda_runtime_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
        "device": "cpu",
        "error": None,
    }

    try:
        import torch  # type: ignore

        probe["torch_available"] = True
        probe["torch_version"] = str(getattr(torch, "__version__", "unknown"))
        probe["cuda_runtime_version"] = getattr(getattr(torch, "version", None), "cuda", None)

        cuda_available = bool(torch.cuda.is_available())
        probe["cuda_available"] = cuda_available
        if cuda_available:
            device_count = int(torch.cuda.device_count())
            probe["cuda_device_count"] = device_count
            names: list[str] = []
            for index in range(device_count):
                try:
                    names.append(str(torch.cuda.get_device_name(index)))
                except Exception:
                    names.append("unknown")
            probe["cuda_device_names"] = names
    except Exception as exc:  # pragma: no cover - depends on runtime image
        probe["error"] = str(exc)

    probe["device"] = "cuda" if probe["cuda_available"] else "cpu"
    return probe


def _print_health_summary(probe: dict[str, Any]) -> None:
    cuda_state = "visible" if probe["cuda_available"] else "not-visible"
    print(
        f"[healthcheck] torch={probe['torch_version']} cuda={cuda_state} "
        f"runtime={probe['cuda_runtime_version']}"
    )
    if probe["cuda_available"] and probe["cuda_device_names"]:
        names = ", ".join(str(name) for name in probe["cuda_device_names"])
        print(f"[healthcheck] devices={names}")
    if probe["error"]:
        print(f"[healthcheck] probe-warning={probe['error']}")
    print(f"[healthcheck] effective-device={probe['device']}")


def _normalize_remainder(values: Sequence[str]) -> list[str]:
    items = list(values)
    if items and items[0] == "--":
        return items[1:]
    return items


def _strip_device_flags(args: Sequence[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    items = list(args)
    while index < len(items):
        token = items[index]
        if token == "--device":
            index += 2
            continue
        if token.startswith("--device="):
            index += 1
            continue
        stripped.append(token)
        index += 1
    return stripped


def _build_runtime_env(probe: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env[FORCED_DEVICE_ENV] = str(probe["device"])
    env[CUDA_VISIBLE_ENV] = "1" if probe["device"] == "cuda" else "0"
    return env


def _release_probe_cuda_cache(probe: dict[str, Any]) -> None:
    if not probe.get("torch_available"):
        return

    try:
        import gc
        import torch  # type: ignore

        if probe.get("cuda_available") and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        gc.collect()
    except Exception:
        # Best-effort cleanup only.
        return


def _run_command(command: list[str], probe: dict[str, Any], require_cuda: bool) -> int:
    _print_health_summary(probe)
    if require_cuda and probe["device"] != "cuda":
        print("[healthcheck] CUDA is required but not visible in this container.", file=sys.stderr)
        return 1

    rendered = " ".join(shlex.quote(part) for part in command)
    print(f"[container-runtime] launching={rendered}")

    # Replace this process with the target app so probe-related CUDA allocations
    # do not remain resident while Streamlit/pipeline is idle.
    env = _build_runtime_env(probe)
    _release_probe_cuda_cache(probe)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.execvpe(command[0], command, env)
    except FileNotFoundError:
        print(f"[container-runtime] command not found: {command[0]}", file=sys.stderr)
        return 127
    except Exception as exc:
        print(f"[container-runtime] failed to launch command: {exc}", file=sys.stderr)
        return 1


def _streamlit_command(address: str, port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app/ui/streamlit_app.py",
        f"--server.address={address}",
        f"--server.port={port}",
    ]


def _pipeline_command(args: Sequence[str], detected_device: str) -> list[str]:
    pipeline_args = _strip_device_flags(_normalize_remainder(args))
    if not pipeline_args:
        pipeline_args = ["--help"]

    return [
        sys.executable,
        "-m",
        "app.main",
        "--device",
        detected_device,
        *pipeline_args,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="container_runtime",
        description="Container helper for CUDA healthcheck and runtime device selection.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    health_parser = subparsers.add_parser("healthcheck", help="Print CUDA visibility and effective device.")
    health_parser.add_argument("--json", action="store_true", help="Print healthcheck payload as JSON")
    health_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero code when CUDA is not visible.",
    )

    pipeline_parser = subparsers.add_parser("pipeline", help="Run app.main with an enforced device.")
    pipeline_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero code when CUDA is not visible.",
    )
    pipeline_parser.add_argument("pipeline_args", nargs=argparse.REMAINDER, help="Arguments forwarded to app.main")

    streamlit_parser = subparsers.add_parser("streamlit", help="Run Streamlit UI with enforced device.")
    streamlit_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero code when CUDA is not visible.",
    )
    streamlit_parser.add_argument("--address", default="0.0.0.0", help="Streamlit bind address")
    streamlit_parser.add_argument("--port", type=int, default=8501, help="Streamlit port")

    run_parser = subparsers.add_parser("run", help="Run an arbitrary command with an enforced device environment.")
    run_parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Return a non-zero code when CUDA is not visible.",
    )
    run_parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to execute")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    probe = detect_runtime_device()

    if args.subcommand == "healthcheck":
        if args.json:
            print(json.dumps(probe, ensure_ascii=True))
        else:
            _print_health_summary(probe)
        _release_probe_cuda_cache(probe)
        if args.require_cuda and probe["device"] != "cuda":
            return 1
        return 0

    if args.subcommand == "pipeline":
        return _run_command(
            _pipeline_command(args.pipeline_args, str(probe["device"])),
            probe,
            bool(args.require_cuda),
        )

    if args.subcommand == "streamlit":
        return _run_command(
            _streamlit_command(args.address, int(args.port)),
            probe,
            bool(args.require_cuda),
        )

    run_command = _normalize_remainder(args.command)
    if not run_command:
        parser.error("run requires a command after '--'")
    return _run_command(run_command, probe, bool(args.require_cuda))


if __name__ == "__main__":
    raise SystemExit(main())