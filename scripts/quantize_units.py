"""Quantize cached SSL features into discrete unit IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fit_kmeans import codebook_path, l2_normalize
from ssl_asr.data.asr_dataset import cache_path_for_row, read_manifest, unit_path_for_row
from ssl_asr.features import load_feature_cache
from ssl_asr.units import deduplicate_units, duration_buckets


def unit_cache_path(
    unit_root: Path,
    row: dict,
    *,
    split: str,
    model_name: str,
    layer: int,
    codebook_size: int,
) -> Path:
    return unit_path_for_row(
        unit_root,
        row,
        split=split,
        model_name=model_name,
        layer=layer,
        codebook_size=codebook_size,
    )


def quantize_manifest(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    feature_root = (root / args.feature_root).resolve()
    unit_root = root / args.unit_root
    split = manifest_path.stem
    rows = read_manifest(manifest_path, limit=args.limit_utts)
    model = joblib.load(
        codebook_path(unit_root, model_name=args.model, layer=args.layer, codebook_size=args.codebook_size)
    )

    for row in rows:
        feature_path = cache_path_for_row(
            feature_root,
            row,
            split=split,
            model_name=args.model,
            layer=args.layer,
        )
        features = load_feature_cache(feature_path).features.float().numpy()
        units = torch.tensor(model.predict(l2_normalize(features)).astype("int64"), dtype=torch.long)
        deduped, lengths = deduplicate_units(units.tolist())
        output_path = unit_cache_path(
            unit_root,
            row,
            split=split,
            model_name=args.model,
            layer=args.layer,
            codebook_size=args.codebook_size,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "utterance_id": row["utterance_id"],
                "units": units,
                "deduplicated_units": torch.tensor(deduped, dtype=torch.long),
                "run_lengths": torch.tensor(lengths, dtype=torch.long),
                "duration_buckets": torch.tensor(duration_buckets(lengths), dtype=torch.long),
                "model": args.model,
                "layer": args.layer,
                "codebook_size": args.codebook_size,
            },
            output_path,
        )
        print(f"[units] wrote {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize cached SSL features with a fitted K-means codebook.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--unit-root", type=Path, default=Path("units"))
    parser.add_argument("--model", default="wavlm-base-plus")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--codebook-size", type=int, default=100)
    parser.add_argument("--limit-utts", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    quantize_manifest(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
