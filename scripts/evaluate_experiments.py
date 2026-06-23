"""Evaluate all trained experiment checkpoints on their configured test manifests."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import yaml


def iter_configs(config_dir: Path, pattern: str | None) -> list[Path]:
    configs = sorted(config_dir.glob("*.yaml"))
    if pattern:
        configs = [path for path in configs if pattern in path.stem]
    return configs


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("[eval-all] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all experiment checkpoints.")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/experiments"))
    parser.add_argument("--pattern")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for config_path in iter_configs(args.config_dir, args.pattern):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        checkpoint = Path(config["output_dir"]) / "best.pt"
        if args.skip_missing and not checkpoint.exists():
            print(f"[skip] missing checkpoint {checkpoint}")
            continue
        for manifest in config["data"].get("test_manifests", []):
            command = [
                sys.executable,
                "scripts/evaluate.py",
                "--checkpoint",
                str(checkpoint),
                "--manifest",
                manifest,
                "--device",
                args.device,
            ]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
