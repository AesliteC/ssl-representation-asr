import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.data.librispeech import iter_transcript_tree, write_manifest


class LibriSpeechManifestTests(unittest.TestCase):
    def test_iter_transcript_tree_reads_librispeech_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "LibriSpeech" / "dev-clean"
            chapter_dir = split_dir / "1272" / "128104"
            chapter_dir.mkdir(parents=True)
            (chapter_dir / "1272-128104-0000.flac").write_bytes(b"fake")
            (chapter_dir / "1272-128104.trans.txt").write_text(
                "1272-128104-0000 Hello, WORLD!\n",
                encoding="utf-8",
            )

            records = list(iter_transcript_tree(split_dir, split="dev-clean"))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].utterance_id, "1272-128104-0000")
        self.assertEqual(records[0].speaker_id, "1272")
        self.assertEqual(records[0].chapter_id, "128104")
        self.assertEqual(records[0].text, "Hello, WORLD!")
        self.assertEqual(records[0].normalized_text, "HELLO WORLD")

    def test_iter_transcript_tree_raises_for_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "dev-clean"
            chapter_dir = split_dir / "1272" / "128104"
            chapter_dir.mkdir(parents=True)
            (chapter_dir / "1272-128104.trans.txt").write_text(
                "1272-128104-0000 HELLO\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "1272-128104-0000.flac"):
                list(iter_transcript_tree(split_dir, split="dev-clean"))

    def test_write_manifest_outputs_jsonl_with_relative_audio_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_dir = root / "dev-clean"
            chapter_dir = split_dir / "1272" / "128104"
            chapter_dir.mkdir(parents=True)
            (chapter_dir / "1272-128104-0000.flac").write_bytes(b"fake")
            (chapter_dir / "1272-128104.trans.txt").write_text(
                "1272-128104-0000 HELLO\n",
                encoding="utf-8",
            )
            records = list(iter_transcript_tree(split_dir, split="dev-clean"))
            manifest_path = root / "manifest.jsonl"

            write_manifest(records, manifest_path, audio_root=root)
            row = json.loads(manifest_path.read_text(encoding="utf-8").strip())

        self.assertEqual(row["audio_filepath"], "dev-clean/1272/128104/1272-128104-0000.flac")
        self.assertEqual(row["text"], "HELLO")
        self.assertEqual(row["normalized_text"], "HELLO")
        self.assertEqual(row["split"], "dev-clean")


if __name__ == "__main__":
    unittest.main()
