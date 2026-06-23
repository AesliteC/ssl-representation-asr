import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.units import deduplicate_units, duration_buckets, fixed_code_bitrate, token_rate


class UnitTests(unittest.TestCase):
    def test_deduplicate_units_returns_runs_and_lengths(self):
        units = [4, 4, 4, 7, 7, 4, 9]

        deduped, lengths = deduplicate_units(units)

        self.assertEqual(deduped, [4, 7, 4, 9])
        self.assertEqual(lengths, [3, 2, 1, 1])

    def test_duration_buckets_use_log2_and_cap_at_seven(self):
        self.assertEqual(duration_buckets([1, 2, 3, 4, 8, 256]), [0, 1, 1, 2, 3, 7])

    def test_token_rate_uses_duration_seconds(self):
        self.assertEqual(token_rate(num_tokens=100, duration_seconds=2.0), 50.0)
        with self.assertRaisesRegex(ValueError, "duration_seconds"):
            token_rate(num_tokens=1, duration_seconds=0)

    def test_fixed_code_bitrate_uses_ceil_log2_codebook_size(self):
        self.assertEqual(fixed_code_bitrate(token_rate_hz=50.0, codebook_size=100), 350.0)
        self.assertEqual(
            fixed_code_bitrate(token_rate_hz=10.0, codebook_size=50, duration_bits=3),
            10.0 * (math.ceil(math.log2(50)) + 3),
        )
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            fixed_code_bitrate(token_rate_hz=1.0, codebook_size=1)


if __name__ == "__main__":
    unittest.main()
