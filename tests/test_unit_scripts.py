import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.fit_kmeans import codebook_path
from scripts.quantize_units import unit_cache_path


class UnitScriptTests(unittest.TestCase):
    def test_codebook_path_uses_model_layer_and_codebook_size(self):
        path = codebook_path(Path("units"), model_name="wavlm-base-plus", layer=9, codebook_size=100)

        self.assertEqual(path.as_posix(), "units/codebooks/wavlm-base-plus/layer9/k100.joblib")

    def test_unit_cache_path_matches_dataset_layout(self):
        row = {"utterance_id": "1272-128104-0000"}

        path = unit_cache_path(
            Path("units"),
            row,
            split="dev-clean",
            model_name="wavlm-base-plus",
            layer=9,
            codebook_size=100,
        )

        self.assertEqual(
            path.as_posix(),
            "units/wavlm-base-plus/layer9/k100/dev-clean/1272/128104/1272-128104-0000.pt",
        )


if __name__ == "__main__":
    unittest.main()
