"""Build JSONL manifests from LibriSpeech-style transcript trees."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_asr.data.librispeech import iter_transcript_tree, write_manifest


DEFAULT_SPLITS = {
    "libri-light-1h": Path("data/raw/librispeech_finetuning/1h"),
    "libri-light-10h": Path("data/raw/librispeech_finetuning"),
    "dev-clean": Path("data/raw/LibriSpeech/dev-clean"),
    "test-clean": Path("data/raw/LibriSpeech/test-clean"),
    "test-other": Path("data/raw/LibriSpeech/test-other"),
    "train-clean-100": Path("data/raw/LibriSpeech/train-clean-100"),
}


def build_manifest(split_dir: Path, output_path: Path, *, split: str, audio_root: Path) -> int:
    records = list(iter_transcript_tree(split_dir, split=split))
    write_manifest(records, output_path, audio_root=audio_root)
    return len(records)


def build_default_manifests(root: Path, output_dir: Path, splits: Sequence[str] | None) -> None:
    selected = tuple(splits or DEFAULT_SPLITS.keys())
    unknown = set(selected) - set(DEFAULT_SPLITS)
    if unknown:
        raise ValueError(f"Unknown default split: {', '.join(sorted(unknown))}")

    for split in selected:
        split_dir = root / DEFAULT_SPLITS[split]
        output_path = output_dir / f"{split}.jsonl"
        if not split_dir.exists():
            print(f"[manifest] skip missing {split}: {split_dir}", file=sys.stderr)
            continue
        count = build_manifest(split_dir, output_path, split=split, audio_root=root)
        print(f"[manifest] wrote {count} records -> {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build ASR JSONL manifests from transcript trees.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--split-dir", type=Path, help="Single transcript tree to scan.")
    parser.add_argument("--split", help="Split name for --split-dir, or default split names.")
    parser.add_argument("--output", type=Path, help="Output JSONL path for --split-dir.")
    args = parser.parse_args(argv)

    if bool(args.split_dir) != bool(args.output):
        parser.error("--split-dir and --output must be provided together")
    if args.split_dir and not args.split:
        parser.error("--split is required when using --split-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve()

    if args.split_dir:
        split_dir = (root / args.split_dir).resolve()
        output_path = (root / args.output).resolve()
        count = build_manifest(split_dir, output_path, split=args.split, audio_root=root)
        print(f"[manifest] wrote {count} records -> {output_path}")
        return 0

    selected_splits = args.split.split(",") if args.split else None
    build_default_manifests(root, output_dir, selected_splits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
