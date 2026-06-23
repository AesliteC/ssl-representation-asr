import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from ssl_asr.data.training import CTCExample, Seq2SeqExample, collate_ctc, collate_seq2seq
from ssl_asr.models.ctc import BiLSTMCTC
from ssl_asr.models.transformer import TransformerASR
from ssl_asr.text import CHAR_SYMBOLS
from scripts.generate_experiment_configs import build_experiment_specs


class TrainingStackTests(unittest.TestCase):
    def test_collate_ctc_pads_features_and_uses_one_based_targets(self):
        examples = [
            CTCExample(
                utterance_id="utt1",
                features=torch.ones(3, 4),
                text="AB",
                normalized_text="AB",
            ),
            CTCExample(
                utterance_id="utt2",
                features=torch.ones(2, 4) * 2,
                text="A",
                normalized_text="A",
            ),
        ]

        batch = collate_ctc(examples)

        self.assertEqual(tuple(batch.features.shape), (2, 3, 4))
        self.assertEqual(batch.feature_lengths.tolist(), [3, 2])
        self.assertEqual(batch.targets.tolist(), [1, 2, 1])
        self.assertEqual(batch.target_lengths.tolist(), [2, 1])
        self.assertEqual(batch.normalized_texts, ["AB", "A"])

    def test_collate_ctc_pads_discrete_unit_sequences(self):
        examples = [
            CTCExample("utt1", torch.tensor([1, 2, 3]), "A", "A"),
            CTCExample("utt2", torch.tensor([4, 5]), "B", "B"),
        ]

        batch = collate_ctc(examples)

        self.assertEqual(tuple(batch.features.shape), (2, 3))
        self.assertEqual(batch.features.dtype, torch.long)
        self.assertEqual(batch.features.tolist(), [[1, 2, 3], [4, 5, 0]])

    def test_bilstm_ctc_outputs_time_major_logits(self):
        model = BiLSTMCTC(input_dim=4, vocab_size=len(CHAR_SYMBOLS) + 1, hidden_size=8)
        features = torch.randn(2, 5, 4)
        lengths = torch.tensor([5, 3])

        logits, output_lengths = model(features, lengths)

        self.assertEqual(tuple(logits.shape), (5, 2, len(CHAR_SYMBOLS) + 1))
        self.assertEqual(output_lengths.tolist(), [5, 3])

    def test_collate_seq2seq_adds_bos_eos_and_padding(self):
        examples = [
            Seq2SeqExample("utt1", torch.ones(3, 4), "AB", "AB"),
            Seq2SeqExample("utt2", torch.ones(2, 4), "A", "A"),
        ]

        batch = collate_seq2seq(examples)

        self.assertEqual(tuple(batch.features.shape), (2, 3, 4))
        self.assertEqual(batch.decoder_inputs.tolist(), [[1, 3, 4], [1, 3, 0]])
        self.assertEqual(batch.labels.tolist(), [[3, 4, 2], [3, 2, 0]])

    def test_transformer_asr_outputs_batch_time_vocab_logits(self):
        model = TransformerASR(
            input_dim=4,
            vocab_size=len(CHAR_SYMBOLS) + 4,
            d_model=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nhead=4,
            dim_feedforward=32,
        )
        features = torch.randn(2, 6, 4)
        feature_lengths = torch.tensor([6, 4])
        decoder_tokens = torch.tensor([[1, 5, 2], [1, 6, 2]])

        logits = model(features, feature_lengths, decoder_tokens)

        self.assertEqual(tuple(logits.shape), (2, 3, len(CHAR_SYMBOLS) + 4))

    def test_transformer_asr_accepts_discrete_unit_inputs(self):
        model = TransformerASR(
            codebook_size=10,
            vocab_size=len(CHAR_SYMBOLS) + 4,
            d_model=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            nhead=4,
            dim_feedforward=32,
        )
        units = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        feature_lengths = torch.tensor([4, 2])
        decoder_tokens = torch.tensor([[1, 5], [1, 6]])

        logits = model(units, feature_lengths, decoder_tokens)

        self.assertEqual(tuple(logits.shape), (2, 2, len(CHAR_SYMBOLS) + 4))

    def test_experiment_matrix_contains_nineteen_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            specs = build_experiment_specs(Path(tmp))

        self.assertEqual(len(specs), 19)
        self.assertEqual(len({spec["name"] for spec in specs}), 19)


if __name__ == "__main__":
    unittest.main()
