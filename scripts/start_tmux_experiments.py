"""Start the full ASR experiment pipeline in a tmux session."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def build_pipeline_command(
    *,
    python_executable: str,
    device: str,
    log_path: Path,
    limit_utts: int | None = None,
    limit_train: int | None = None,
    limit_dev: int | None = None,
) -> str:
    prepare = [
        python_executable,
        "scripts/prepare_experiment_assets.py",
        "--device",
        device,
    ]
    if limit_utts is not None:
        prepare += ["--limit-utts", str(limit_utts)]

    train = [
        python_executable,
        "scripts/run_experiments.py",
        "--config-dir",
        "configs/experiments",
        "--device",
        device,
        "--skip-existing",
    ]
    if limit_train is not None:
        train += ["--limit-train", str(limit_train)]
    if limit_dev is not None:
        train += ["--limit-dev", str(limit_dev)]

    evaluate = [
        python_executable,
        "scripts/evaluate_experiments.py",
        "--config-dir",
        "configs/experiments",
        "--device",
        device,
        "--skip-missing",
    ]

    summarize = [
        python_executable,
        "scripts/summarize.py",
        "--outputs-dir",
        "outputs",
        "--output",
        "results/summary.csv",
    ]

    steps = [
        "mkdir -p logs",
        " ".join(prepare),
        " ".join(train),
        " ".join(evaluate),
        " ".join(summarize),
    ]
    body = " && ".join(steps)
    return f"({body}) 2>&1 | tee -a {log_path.as_posix()}"


def build_powershell_pipeline_command(
    *,
    script_path: Path,
    device: str,
    limit_utts: int | None = None,
    limit_train: int | None = None,
    limit_dev: int | None = None,
) -> str:
    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        script_path.as_posix(),
        "-Device",
        device,
    ]
    if limit_utts is not None:
        command += ["-LimitUtterances", str(limit_utts)]
    if limit_train is not None:
        command += ["-LimitTrain", str(limit_train)]
    if limit_dev is not None:
        command += ["-LimitDev", str(limit_dev)]
    return " ".join(command)


def build_tmux_command(*, tmux_executable: str, session: str, pipeline_command: str) -> list[str]:
    return [tmux_executable, "new-session", "-d", "-s", session, pipeline_command]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start full ASR experiments in tmux.")
    parser.add_argument("--session", default="ssl_asr_full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tmux", default="tmux")
    parser.add_argument("--windows-powershell", action="store_true", default=True)
    parser.add_argument("--pipeline-script", type=Path, default=Path("scripts/run_full_experiments.ps1"))
    parser.add_argument("--log", type=Path, default=Path("logs/full_experiments.log"))
    parser.add_argument("--limit-utts", type=int, help="Smoke mode: limit asset preparation.")
    parser.add_argument("--limit-train", type=int, help="Smoke mode: limit training utterances.")
    parser.add_argument("--limit-dev", type=int, help="Smoke mode: limit dev utterances.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.windows_powershell:
        pipeline = build_powershell_pipeline_command(
            script_path=args.pipeline_script,
            device=args.device,
            limit_utts=args.limit_utts,
            limit_train=args.limit_train,
            limit_dev=args.limit_dev,
        )
    else:
        pipeline = build_pipeline_command(
            python_executable=args.python,
            device=args.device,
            log_path=args.log,
            limit_utts=args.limit_utts,
            limit_train=args.limit_train,
            limit_dev=args.limit_dev,
        )
    command = build_tmux_command(tmux_executable=args.tmux, session=args.session, pipeline_command=pipeline)
    if args.dry_run:
        print(" ".join(command))
        return 0
    if shutil.which(args.tmux) is None and not Path(args.tmux).exists():
        raise RuntimeError(f"tmux was not found: {args.tmux}")
    subprocess.run(command, check=True)
    print(f"[tmux] started session {args.session}")
    print(f"[tmux] attach with: tmux attach -t {args.session}")
    print(f"[tmux] log: {args.log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
