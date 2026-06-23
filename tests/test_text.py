import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ssl_asr.text import CHAR_SYMBOLS, Vocabulary, normalize_text


class TextTests(unittest.TestCase):
    def test_normalize_text_uppercases_keeps_apostrophe_and_collapses_spaces(self):
        text = "  Hello, world! it's   ASR-time.  "

        self.assertEqual(normalize_text(text), "HELLO WORLD IT'S ASRTIME")

    def test_vocabulary_round_trip(self):
        vocab = Vocabulary.from_symbols(CHAR_SYMBOLS)
        encoded = vocab.encode("A B'")

        self.assertEqual(vocab.decode(encoded), "A B'")
        self.assertEqual(vocab.token_to_id["A"], 0)
        self.assertEqual(vocab.token_to_id[" "], len(CHAR_SYMBOLS) - 1)

    def test_vocabulary_rejects_unknown_character(self):
        vocab = Vocabulary.from_symbols(CHAR_SYMBOLS)

        with self.assertRaisesRegex(ValueError, "Unsupported character"):
            vocab.encode("HELLO 1")


if __name__ == "__main__":
    unittest.main()
