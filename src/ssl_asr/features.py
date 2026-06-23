"""Frozen SSL feature cache helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


@dataclass(frozen=True)
class FeatureCacheItem:
    utterance_id: str
    features: torch.Tensor
    layer: int
    model_id: str
    revision: str
    sample_rate: int


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def select_hidden_state(hidden_states: Sequence[torch.Tensor], *, layer: int) -> torch.Tensor:
    if layer < 0 or layer >= len(hidden_states):
        raise ValueError(f"layer must be between 0 and {len(hidden_states) - 1}, got {layer}")
    return hidden_states[layer]


def save_feature_cache(item: FeatureCacheItem, cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "utterance_id": item.utterance_id,
        "features": item.features.detach().cpu().to(torch.float16),
        "layer": item.layer,
        "model_id": item.model_id,
        "revision": item.revision,
        "sample_rate": item.sample_rate,
    }
    torch.save(payload, cache_path)


def load_feature_cache(cache_path: Path) -> FeatureCacheItem:
    payload = torch.load(cache_path, map_location="cpu")
    return FeatureCacheItem(
        utterance_id=payload["utterance_id"],
        features=payload["features"],
        layer=payload["layer"],
        model_id=payload["model_id"],
        revision=payload["revision"],
        sample_rate=payload["sample_rate"],
    )


def utterance_cache_path(cache_root: Path, utterance_id: str) -> Path:
    parts = utterance_id.split("-")
    if len(parts) >= 3:
        return cache_root / parts[0] / parts[1] / f"{utterance_id}.pt"
    return cache_root / f"{utterance_id}.pt"
