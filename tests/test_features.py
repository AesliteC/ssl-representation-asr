import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.features import (
    FeatureCacheItem,
    freeze_module,
    load_feature_cache,
    save_feature_cache,
    select_hidden_state,
    utterance_cache_path,
)


class FeatureTests(unittest.TestCase):
    def test_freeze_module_disables_grad_and_eval_mode(self):
        module = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Dropout())
        module.train()

        result = freeze_module(module)

        self.assertIs(result, module)
        self.assertFalse(module.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in module.parameters()))

    def test_select_hidden_state_uses_one_based_transformer_layer_index(self):
        hidden_states = tuple(torch.full((1, 2, 3), value, dtype=torch.float32) for value in range(13))

        selected = select_hidden_state(hidden_states, layer=9)

        self.assertTrue(torch.equal(selected, hidden_states[9]))
        with self.assertRaisesRegex(ValueError, "layer must be between"):
            select_hidden_state(hidden_states, layer=13)

    def test_feature_cache_round_trip_uses_fp16_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "utt.pt"
            item = FeatureCacheItem(
                utterance_id="utt",
                features=torch.ones(2, 3, dtype=torch.float32),
                layer=9,
                model_id="microsoft/wavlm-base-plus",
                revision="abc123",
                sample_rate=16_000,
            )

            save_feature_cache(item, cache_path)
            loaded = load_feature_cache(cache_path)

        self.assertEqual(loaded.utterance_id, "utt")
        self.assertEqual(loaded.features.dtype, torch.float16)
        self.assertEqual(tuple(loaded.features.shape), (2, 3))
        self.assertEqual(loaded.layer, 9)
        self.assertEqual(loaded.revision, "abc123")

    def test_utterance_cache_path_shards_by_utterance_id(self):
        path = utterance_cache_path(Path("features"), "1272-128104-0000")

        self.assertEqual(path.as_posix(), "features/1272/128104/1272-128104-0000.pt")


if __name__ == "__main__":
    unittest.main()
