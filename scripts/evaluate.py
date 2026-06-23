"""Evaluate a trained ASR checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train import build_dataloader, build_model, evaluate_ctc, evaluate_transformer


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    manifest: Path,
    root: Path,
    device_name: str,
    limit: int | None,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device(device_name)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    loader = build_dataloader(manifest, config=config, root=root, limit=limit, shuffle=False)
    if config["model"]["type"] == "ctc":
        metrics = evaluate_ctc(model, loader, device)
    else:
        metrics = evaluate_transformer(model, loader, device)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one ASR checkpoint on one manifest.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    manifest = (root / args.manifest).resolve()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        manifest=manifest,
        root=root,
        device_name=args.device,
        limit=args.limit,
    )
    output = args.output or args.checkpoint.parent / "eval" / f"{manifest.stem}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[eval] {manifest.stem} wer={metrics['wer']:.4f} cer={metrics['cer']:.4f} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
