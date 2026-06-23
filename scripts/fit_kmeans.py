"""Fit MiniBatchKMeans codebooks from cached SSL features."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_asr.data.asr_dataset import cache_path_for_row, read_manifest
from ssl_asr.features import load_feature_cache
from ssl_asr.kmeans import fit_kmeans, sample_frames


def codebook_path(unit_root: Path, *, model_name: str, layer: int, codebook_size: int) -> Path:
    return unit_root / "codebooks" / model_name / f"layer{layer}" / f"k{codebook_size}.joblib"


def l2_normalize(frames: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(frames, axis=1, keepdims=True)
    return frames / np.maximum(norms, eps)


def load_feature_arrays(
    manifest_path: Path,
    *,
    feature_root: Path,
    model_name: str,
    layer: int,
    limit: int | None,
) -> list[np.ndarray]:
    rows = read_manifest(manifest_path, limit=limit)
    split = manifest_path.stem
    arrays = []
    for row in rows:
        path = cache_path_for_row(feature_root, row, split=split, model_name=model_name, layer=layer)
        arrays.append(load_feature_cache(path).features.float().numpy())
    return arrays


def fit_codebook(args: argparse.Namespace) -> Path:
    root = args.root.resolve()
    manifest_path = (root / args.manifest).resolve()
    feature_root = (root / args.feature_root).resolve()
    unit_root = root / args.unit_root
    arrays = load_feature_arrays(
        manifest_path,
        feature_root=feature_root,
        model_name=args.model,
        layer=args.layer,
        limit=args.limit_utts,
    )
    frames = sample_frames(arrays, max_frames=args.max_frames, seed=args.seed)
    frames = l2_normalize(frames.astype(np.float32))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="MiniBatchKMeans is known to have a memory leak")
        model = fit_kmeans(
            frames,
            codebook_size=args.codebook_size,
            seed=args.seed,
            batch_size=args.batch_size,
            max_iter=args.max_iter,
            n_init=args.n_init,
        )
    output_path = codebook_path(
        unit_root,
        model_name=args.model,
        layer=args.layer,
        codebook_size=args.codebook_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[kmeans] wrote {output_path}")
    return output_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a K-means codebook from cached SSL features.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=Path("features"))
    parser.add_argument("--unit-root", type=Path, default=Path("units"))
    parser.add_argument("--model", default="wavlm-base-plus")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--codebook-size", type=int, default=100)
    parser.add_argument("--max-frames", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-utts", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    fit_codebook(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
