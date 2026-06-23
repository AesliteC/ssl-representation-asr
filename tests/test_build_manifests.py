import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_manifests import build_manifest


class BuildManifestsTests(unittest.TestCase):
    def test_build_manifest_writes_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "data" / "raw" / "LibriSpeech" / "dev-clean"
            chapter_dir = split_dir / "1272" / "128104"
            chapter_dir.mkdir(parents=True)
            (chapter_dir / "1272-128104-0000.flac").write_bytes(b"fake")
            (chapter_dir / "1272-128104.trans.txt").write_text(
                "1272-128104-0000 Hello!\n",
                encoding="utf-8",
            )
            output_path = root / "data" / "manifests" / "dev-clean.jsonl"

            count = build_manifest(split_dir, output_path, split="dev-clean", audio_root=root)
            row = json.loads(output_path.read_text(encoding="utf-8").strip())

        self.assertEqual(count, 1)
        self.assertEqual(
            row["audio_filepath"],
            "data/raw/LibriSpeech/dev-clean/1272/128104/1272-128104-0000.flac",
        )
        self.assertEqual(row["normalized_text"], "HELLO")


if __name__ == "__main__":
    unittest.main()
