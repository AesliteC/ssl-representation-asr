"""Prepare manifests, SSL features, K-means codebooks, and unit caches for all experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


WAVLM_MODEL = "wavlm-base-plus"
SSL_COMPARISON_MODELS = ("hubert-base-ls960", "wav2vec2-base")
MAIN_MANIFESTS = (
    "data/manifests/libri-light-10h.jsonl",
    "data/manifests/dev-clean.jsonl",
    "data/manifests/test-clean.jsonl",
    "data/manifests/test-other.jsonl",
)
SCALE_MANIFESTS = (
    "data/manifests/libri-light-1h.jsonl",
    "data/manifests/train-clean-100.jsonl",
)


def command_list(args: argparse.Namespace) -> list[list[str]]:
    commands: list[list[str]] = [[sys.executable, "scripts/build_manifests.py"]]

    for layer in (3, 6, 9, 12):
        for manifest in MAIN_MANIFESTS:
            commands.append(extract_command(args, manifest, WAVLM_MODEL, layer))

    for manifest in SCALE_MANIFESTS:
        commands.append(extract_command(args, manifest, WAVLM_MODEL, 9))

    for model in SSL_COMPARISON_MODELS:
        for manifest in MAIN_MANIFESTS:
            commands.append(extract_command(args, manifest, model, 9))

    for codebook_size in (50, 100, 200):
        commands.append(
            [
                sys.executable,
                "scripts/fit_kmeans.py",
                "--manifest",
                "data/manifests/libri-light-10h.jsonl",
                "--model",
                WAVLM_MODEL,
                "--layer",
                "9",
                "--codebook-size",
                str(codebook_size),
            ]
            + optional_limit(args, "--limit-utts")
        )
        for manifest in (*MAIN_MANIFESTS, *SCALE_MANIFESTS):
            commands.append(
                [
                    sys.executable,
                    "scripts/quantize_units.py",
                    "--manifest",
                    manifest,
                    "--model",
                    WAVLM_MODEL,
                    "--layer",
                    "9",
                    "--codebook-size",
                    str(codebook_size),
                ]
                + optional_limit(args, "--limit-utts")
            )
    return commands


def extract_command(args: argparse.Namespace, manifest: str, model: str, layer: int) -> list[str]:
    command = [
        sys.executable,
        "scripts/extract_features.py",
        "--manifest",
        manifest,
        "--model",
        model,
        "--layer",
        str(layer),
        "--device",
        args.device,
    ]
    if args.limit_utts is not None:
        command += ["--limit", str(args.limit_utts)]
    return command


def optional_limit(args: argparse.Namespace, flag: str) -> list[str]:
    return [flag, str(args.limit_utts)] if args.limit_utts is not None else []


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("[prepare] " + " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached assets needed by the full experiment matrix.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit-utts", type=int, help="Smoke mode: only process first N utterances per manifest.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for command in command_list(args):
        run_command(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
