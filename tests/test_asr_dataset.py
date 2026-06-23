import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.data.asr_dataset import ASRDataset, cache_path_for_row
from ssl_asr.features import FeatureCacheItem, save_feature_cache


class ASRDatasetTests(unittest.TestCase):
    def test_cache_path_for_row_matches_feature_layout(self):
        row = {"utterance_id": "1272-128104-0000"}

        path = cache_path_for_row(
            Path("features"),
            row,
            split="dev-clean",
            model_name="wavlm-base-plus",
            layer=9,
        )

        self.assertEqual(
            path.as_posix(),
            "features/wavlm-base-plus/layer9/dev-clean/1272/128104/1272-128104-0000.pt",
        )

    def test_continuous_dataset_loads_feature_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "dev-clean.jsonl"
            row = {
                "utterance_id": "1272-128104-0000",
                "audio_filepath": "unused.flac",
                "text": "Hello",
                "normalized_text": "HELLO",
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            cache_path = cache_path_for_row(
                root / "features",
                row,
                split="dev-clean",
                model_name="wavlm-base-plus",
                layer=9,
            )
            save_feature_cache(
                FeatureCacheItem(
                    utterance_id=row["utterance_id"],
                    features=torch.ones(4, 3),
                    layer=9,
                    model_id="microsoft/wavlm-base-plus",
                    revision="abc",
                    sample_rate=16_000,
                ),
                cache_path,
            )

            dataset = ASRDataset(
                manifest,
                representation={
                    "kind": "continuous",
                    "feature_root": str(root / "features"),
                    "model": "wavlm-base-plus",
                    "layer": 9,
                },
            )
            example = dataset[0]

        self.assertEqual(example.utterance_id, row["utterance_id"])
        self.assertEqual(example.normalized_text, "HELLO")
        self.assertEqual(tuple(example.features.shape), (4, 3))


if __name__ == "__main__":
    unittest.main()
