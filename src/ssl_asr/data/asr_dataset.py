"""ASR datasets backed by manifests and cached representations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset

from ssl_asr.data.training import CTCExample
from ssl_asr.features import load_feature_cache, utterance_cache_path
from ssl_asr.units import deduplicate_units, duration_buckets


def read_manifest(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def cache_path_for_row(
    feature_root: Path,
    row: dict[str, Any],
    *,
    split: str,
    model_name: str,
    layer: int,
) -> Path:
    return utterance_cache_path(feature_root / model_name / f"layer{layer}" / split, row["utterance_id"])


def unit_path_for_row(
    unit_root: Path,
    row: dict[str, Any],
    *,
    split: str,
    model_name: str,
    layer: int,
    codebook_size: int,
) -> Path:
    return utterance_cache_path(
        unit_root / model_name / f"layer{layer}" / f"k{codebook_size}" / split,
        row["utterance_id"],
    )


class ASRDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        *,
        representation: dict[str, Any],
        root: Path | None = None,
        limit: int | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = Path(root or ".")
        self.representation = representation
        self.rows = read_manifest(self.manifest_path, limit=limit)
        self.split = self.manifest_path.stem
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=400,
            hop_length=320,
            n_mels=80,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> CTCExample:
        row = self.rows[index]
        return CTCExample(
            utterance_id=row["utterance_id"],
            features=self.load_features(row),
            text=row.get("text", row.get("normalized_text", "")),
            normalized_text=row["normalized_text"],
        )

    def load_features(self, row: dict[str, Any]) -> torch.Tensor:
        kind = self.representation["kind"]
        if kind == "continuous":
            feature_root = self.root / self.representation.get("feature_root", "features")
            path = cache_path_for_row(
                feature_root,
                row,
                split=self.split,
                model_name=self.representation["model"],
                layer=int(self.representation["layer"]),
            )
            return load_feature_cache(path).features.float()
        if kind == "discrete":
            unit_root = self.root / self.representation.get("unit_root", "units")
            path = unit_path_for_row(
                unit_root,
                row,
                split=self.split,
                model_name=self.representation.get("model", "wavlm-base-plus"),
                layer=int(self.representation.get("layer", 9)),
                codebook_size=int(self.representation["codebook_size"]),
            )
            payload = torch.load(path, map_location="cpu")
            units = payload["units"].long()
            if self.representation.get("deduplicate"):
                deduped, lengths = deduplicate_units(units.tolist())
                units = torch.tensor(deduped, dtype=torch.long)
                if self.representation.get("use_duration"):
                    buckets = torch.tensor(duration_buckets(lengths), dtype=torch.long)
                    units = units * 8 + buckets
            return units
        if kind == "logmel":
            audio_path = self.root / row["audio_filepath"]
            waveform, sample_rate = load_audio(audio_path)
            if sample_rate != 16_000:
                waveform = AF.resample(waveform, sample_rate, 16_000)
            features = torch.log1p(self.mel(waveform).transpose(0, 1))
            return features.float()
        raise ValueError(f"Unsupported representation kind: {kind}")


def load_audio(audio_path: Path) -> tuple[torch.Tensor, int]:
    samples, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(samples).mean(dim=1)
    return waveform, sample_rate
