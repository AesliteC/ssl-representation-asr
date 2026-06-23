import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import download_assets


class DownloadAssetsTests(unittest.TestCase):
    def test_data_archive_manifest_covers_required_splits(self):
        names = {asset.name for asset in download_assets.DATA_ARCHIVES}

        self.assertEqual(
            names,
            {
                "librispeech_finetuning",
                "dev-clean",
                "test-clean",
                "test-other",
                "train-clean-100",
            },
        )

        checksummed = {
            asset.name: asset.md5
            for asset in download_assets.DATA_ARCHIVES
            if asset.md5 is not None
        }
        self.assertEqual(checksummed["dev-clean"], "42e2234ba48799c1f50f24a7926300a1")
        self.assertEqual(checksummed["test-clean"], "32fa31d27d2e1cad72775fee3f4849a9")
        self.assertEqual(checksummed["test-other"], "fb5a50374b501bb3bac4815ee91d3135")
        self.assertEqual(
            checksummed["train-clean-100"], "2a93770f6d5c6c964bc36631d331a522"
        )

    def test_model_manifest_pins_revisions_and_files(self):
        model_ids = {model.model_id for model in download_assets.MODEL_SPECS}

        self.assertEqual(
            model_ids,
            {
                "microsoft/wavlm-base-plus",
                "facebook/hubert-base-ls960",
                "facebook/wav2vec2-base",
            },
        )
        for model in download_assets.MODEL_SPECS:
            self.assertRegex(model.revision, r"^[0-9a-f]{40}$")
            self.assertIn("config.json", model.allow_patterns)
            self.assertTrue(
                "pytorch_model.bin" in model.allow_patterns
                or "model.safetensors" in model.allow_patterns
            )

    def test_select_by_name_filters_data_and_reports_unknowns(self):
        selected = download_assets.select_by_name(
            download_assets.DATA_ARCHIVES, ["dev-clean", "test-clean"]
        )

        self.assertEqual([asset.name for asset in selected], ["dev-clean", "test-clean"])
        with self.assertRaisesRegex(ValueError, "not-a-split"):
            download_assets.select_by_name(download_assets.DATA_ARCHIVES, ["not-a-split"])

    def test_default_network_settings_do_not_hardcode_machine_proxy(self):
        source = Path(download_assets.__file__).read_text(encoding="utf-8")

        blocked_markers = (".".join(["127", "0", "0", "1"]), "78" + "91", "78" + "92")
        for marker in blocked_markers:
            self.assertNotIn(marker, source)
        self.assertEqual(download_assets.build_proxies(force_direct=True), {})
        self.assertIsNone(download_assets.build_proxies(force_direct=False))

    def test_md5_file_calculates_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("asr-downloads\n", encoding="utf-8")

            digest = download_assets.md5_file(path)

        self.assertEqual(digest, "b9bb0b25752a64394a422e911eaa031a")

    def test_safe_extract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "bad.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"nope"
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))

            with tarfile.open(archive_path, "r:gz") as archive:
                with self.assertRaisesRegex(ValueError, "Unsafe archive member"):
                    download_assets.safe_extract(archive, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
