import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.metrics import char_error_rate, edit_distance, word_error_rate


class MetricsTests(unittest.TestCase):
    def test_edit_distance_counts_insertions_deletions_and_substitutions(self):
        self.assertEqual(edit_distance(["A", "B", "C"], ["A", "X", "C", "D"]), 2)

    def test_word_error_rate_uses_reference_word_count(self):
        self.assertEqual(word_error_rate("HELLO WORLD", "HELLO WORD"), 0.5)

    def test_char_error_rate_uses_reference_character_count_without_spaces(self):
        self.assertAlmostEqual(char_error_rate("AB C", "ADC"), 1 / 3)

    def test_error_rates_handle_empty_reference(self):
        self.assertEqual(word_error_rate("", ""), 0.0)
        self.assertEqual(char_error_rate("", "A"), 1.0)


if __name__ == "__main__":
    unittest.main()
