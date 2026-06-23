import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.extract_features import feature_cache_path, read_manifest_rows


class ExtractFeaturesScriptTests(unittest.TestCase):
    def test_read_manifest_rows_respects_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.jsonl"
            rows = [
                {"utterance_id": "utt-1", "audio_filepath": "a.flac"},
                {"utterance_id": "utt-2", "audio_filepath": "b.flac"},
            ]
            manifest_path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )

            loaded = list(read_manifest_rows(manifest_path, limit=1))

        self.assertEqual(loaded, [rows[0]])

    def test_feature_cache_path_uses_manifest_stem_and_layer(self):
        path = feature_cache_path(
            Path("features"),
            manifest_path=Path("data/manifests/dev-clean.jsonl"),
            model_name="wavlm-base-plus",
            layer=9,
            utterance_id="1272-128104-0000",
        )

        self.assertEqual(
            path.as_posix(),
            "features/wavlm-base-plus/layer9/dev-clean/1272/128104/1272-128104-0000.pt",
        )


if __name__ == "__main__":
    unittest.main()
