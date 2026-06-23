import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.ctc import greedy_ctc_decode


class CTCTests(unittest.TestCase):
    def test_greedy_ctc_decode_collapses_repeats_and_removes_blank(self):
        id_to_token = {1: "A", 2: "B", 3: " "}
        token_ids = [0, 1, 1, 0, 2, 2, 3, 3, 0, 1]

        self.assertEqual(greedy_ctc_decode(token_ids, id_to_token, blank_id=0), "AB A")

    def test_greedy_ctc_decode_requires_known_tokens(self):
        with self.assertRaisesRegex(KeyError, "Unknown token id"):
            greedy_ctc_decode([1], {}, blank_id=0)


if __name__ == "__main__":
    unittest.main()
