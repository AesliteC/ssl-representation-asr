"""Run generated ASR experiment configs sequentially."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def iter_configs(config_dir: Path, pattern: str | None) -> list[Path]:
    configs = sorted(config_dir.glob("*.yaml"))
    if pattern:
        configs = [path for path in configs if pattern in path.stem]
    return configs


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("[run] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ASR experiment configs.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--pattern", help="Only run configs whose filename contains this text.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for config_path in iter_configs(args.config_dir, args.pattern):
        output_dir = Path("outputs") / config_path.stem
        if args.skip_existing and (output_dir / "best.pt").exists():
            print(f"[skip] {config_path}")
            continue
        command = [
            sys.executable,
            "scripts/train.py",
            "--config",
            str(config_path),
            "--device",
            args.device,
        ]
        if args.limit_train is not None:
            command += ["--limit-train", str(args.limit_train)]
        if args.limit_dev is not None:
            command += ["--limit-dev", str(args.limit_dev)]
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
