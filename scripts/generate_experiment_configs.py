"""Generate YAML configs for the planned ASR experiment matrix."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import yaml


def base_config(name: str) -> dict:
    return {
        "name": name,
        "seed": 42,
        "data": {
            "train_manifest": "data/manifests/libri-light-10h.jsonl",
            "dev_manifest": "data/manifests/dev-clean.jsonl",
            "test_manifests": [
                "data/manifests/test-clean.jsonl",
                "data/manifests/test-other.jsonl",
            ],
        },
        "representation": {
            "kind": "continuous",
            "model": "wavlm-base-plus",
            "layer": 9,
            "feature_root": "features",
            "input_dim": 768,
        },
        "model": {
            "type": "ctc",
            "hidden_size": 256,
            "projection_dim": 256,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "training": {
            "epochs": 50,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.01,
            "grad_clip": 5.0,
            "patience": 8,
            "num_workers": 0,
        },
        "output_dir": f"outputs/{name}",
    }


def with_updates(config: dict, **updates) -> dict:
    copied = deepcopy(config)
    for path, value in updates.items():
        target = copied
        keys = path.split("__")
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return copied


def build_experiment_specs(root: Path | None = None) -> list[dict]:
    specs = []

    specs.append(with_updates(base_config("baseline_logmel_ctc"), representation__kind="logmel", representation__input_dim=80))
    specs.append(
        with_updates(
            base_config("baseline_logmel_transformer"),
            representation__kind="logmel",
            representation__input_dim=80,
            model__type="transformer",
            training__learning_rate=0.0003,
        )
    )

    for layer in (3, 6, 9, 12):
        specs.append(with_updates(base_config(f"wavlm_layer{layer}_ctc"), representation__layer=layer))

    for model_name in ("hubert-base-ls960", "wav2vec2-base"):
        specs.append(
            with_updates(
                base_config(f"{model_name}_layer9_ctc"),
                representation__model=model_name,
                representation__layer=9,
            )
        )

    for codebook_size in (50, 100, 200):
        specs.append(
            with_updates(
                base_config(f"wavlm_units_k{codebook_size}_ctc"),
                representation__kind="discrete",
                representation__codebook_size=codebook_size,
                representation__unit_root="units",
            )
        )

    specs.append(
        with_updates(
            base_config("wavlm_units_k100_dedup_ctc"),
            representation__kind="discrete",
            representation__codebook_size=100,
            representation__unit_root="units",
            representation__deduplicate=True,
        )
    )
    specs.append(
        with_updates(
            base_config("wavlm_units_k100_dedup_duration_ctc"),
            representation__kind="discrete",
            representation__codebook_size=100,
            representation__unit_root="units",
            representation__deduplicate=True,
            representation__use_duration=True,
        )
    )

    for split_name, train_manifest in (
        ("1h", "data/manifests/libri-light-1h.jsonl"),
        ("100h", "data/manifests/train-clean-100.jsonl"),
    ):
        specs.append(
            with_updates(
                base_config(f"scale_{split_name}_continuous_ctc"),
                data__train_manifest=train_manifest,
            )
        )
        specs.append(
            with_updates(
                base_config(f"scale_{split_name}_units_k100_ctc"),
                data__train_manifest=train_manifest,
                representation__kind="discrete",
                representation__codebook_size=100,
                representation__unit_root="units",
            )
        )

    specs.append(
        with_updates(
            base_config("wavlm_continuous_transformer"),
            model__type="transformer",
            training__learning_rate=0.0003,
        )
    )
    specs.append(
        with_updates(
            base_config("wavlm_units_k100_transformer"),
            representation__kind="discrete",
            representation__codebook_size=100,
            representation__unit_root="units",
            model__type="transformer",
            training__learning_rate=0.0003,
        )
    )
    return specs


def write_configs(configs: Sequence[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for config in configs:
        path = output_dir / f"{config['name']}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate all planned ASR experiment configs.")
    parser.add_argument("--output-dir", type=Path, default=Path("configs/experiments"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configs = build_experiment_specs()
    write_configs(configs, args.output_dir)
    print(f"[configs] wrote {len(configs)} configs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
