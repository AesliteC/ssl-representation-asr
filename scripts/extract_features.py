"""Extract frozen SSL hidden states from ASR manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Sequence

import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoFeatureExtractor, AutoModel


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_asr.features import (
    FeatureCacheItem,
    freeze_module,
    save_feature_cache,
    select_hidden_state,
    utterance_cache_path,
)


DEFAULT_MODELS = {
    "wavlm-base-plus": {
        "path": Path("models/wavlm-base-plus"),
        "model_id": "microsoft/wavlm-base-plus",
        "revision": "4c66d4806a428f2e922ccfa1a962776e232d487b",
    },
    "hubert-base-ls960": {
        "path": Path("models/hubert-base-ls960"),
        "model_id": "facebook/hubert-base-ls960",
        "revision": "dba3bb02fda4248b6e082697eee756de8fe8aa8a",
    },
    "wav2vec2-base": {
        "path": Path("models/wav2vec2-base"),
        "model_id": "facebook/wav2vec2-base",
        "revision": "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
    },
}


def read_manifest_rows(manifest_path: Path, *, limit: int | None = None) -> Iterator[dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                yield json.loads(line)


def feature_cache_path(
    cache_root: Path,
    *,
    manifest_path: Path,
    model_name: str,
    layer: int,
    utterance_id: str,
) -> Path:
    split = manifest_path.stem
    return utterance_cache_path(cache_root / model_name / f"layer{layer}" / split, utterance_id)


def load_audio(audio_path: Path, *, target_sample_rate: int = 16_000) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(dim=1)
    if sample_rate != target_sample_rate:
        waveform = AF.resample(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    return waveform, sample_rate


def extract_manifest(args: argparse.Namespace) -> None:
    model_info = DEFAULT_MODELS[args.model]
    model_path = (args.root / (args.model_path or model_info["path"])).resolve()
    manifest_path = (args.root / args.manifest).resolve()
    cache_root = (args.root / args.output_dir).resolve()
    device = torch.device(args.device)

    processor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True, output_hidden_states=True)
    model = freeze_module(model).to(device)

    for row in read_manifest_rows(manifest_path, limit=args.limit):
        utterance_id = row["utterance_id"]
        output_path = feature_cache_path(
            cache_root,
            manifest_path=manifest_path,
            model_name=args.model,
            layer=args.layer,
            utterance_id=utterance_id,
        )
        if output_path.exists() and not args.force:
            print(f"[features] reuse {output_path}")
            continue

        audio_path = (args.root / row["audio_filepath"]).resolve()
        waveform, sample_rate = load_audio(audio_path)
        inputs = processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            features = select_hidden_state(outputs.hidden_states, layer=args.layer).squeeze(0)

        save_feature_cache(
            FeatureCacheItem(
                utterance_id=utterance_id,
                features=features,
                layer=args.layer,
                model_id=model_info["model_id"],
                revision=model_info["revision"],
                sample_rate=sample_rate,
            ),
            output_path,
        )
        print(f"[features] wrote {output_path}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen SSL hidden-state features for one manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--manifest", type=Path, required=True, help="Input JSONL manifest.")
    parser.add_argument("--model", choices=sorted(DEFAULT_MODELS), default="wavlm-base-plus")
    parser.add_argument("--model-path", type=Path, help="Override local model directory.")
    parser.add_argument("--layer", type=int, default=9, help="Hidden-state layer index.")
    parser.add_argument("--output-dir", type=Path, default=Path("features"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="Extract only the first N rows.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.root = args.root.resolve()
    extract_manifest(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
