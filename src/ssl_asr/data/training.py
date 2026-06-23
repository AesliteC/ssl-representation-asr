"""Training data structures and collation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ssl_asr.text import CHAR_SYMBOLS, Vocabulary


@dataclass(frozen=True)
class CTCExample:
    utterance_id: str
    features: torch.Tensor
    text: str
    normalized_text: str


@dataclass(frozen=True)
class CTCBatch:
    utterance_ids: list[str]
    features: torch.Tensor
    feature_lengths: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    texts: list[str]
    normalized_texts: list[str]


@dataclass(frozen=True)
class Seq2SeqExample:
    utterance_id: str
    features: torch.Tensor
    text: str
    normalized_text: str


@dataclass(frozen=True)
class Seq2SeqBatch:
    utterance_ids: list[str]
    features: torch.Tensor
    feature_lengths: torch.Tensor
    decoder_inputs: torch.Tensor
    labels: torch.Tensor
    texts: list[str]
    normalized_texts: list[str]


def collate_ctc(examples: list[CTCExample]) -> CTCBatch:
    if not examples:
        raise ValueError("examples must not be empty")

    vocab = Vocabulary.from_symbols(CHAR_SYMBOLS)
    feature_lengths = torch.tensor([example.features.shape[0] for example in examples], dtype=torch.long)
    feature_rank = examples[0].features.ndim
    max_frames = int(feature_lengths.max().item())
    if feature_rank == 1:
        padded = torch.zeros(len(examples), max_frames, dtype=torch.long)
    elif feature_rank == 2:
        feature_dim = examples[0].features.shape[-1]
        padded = torch.zeros(len(examples), max_frames, feature_dim, dtype=torch.float32)
    else:
        raise ValueError(f"features must be 1-D or 2-D, got shape {tuple(examples[0].features.shape)}")

    targets = []
    target_lengths = []
    for index, example in enumerate(examples):
        features = example.features
        if features.ndim != feature_rank:
            raise ValueError("all examples in a batch must have the same feature rank")
        if feature_rank == 2:
            if features.shape[-1] != feature_dim:
                raise ValueError("all examples in a batch must have the same feature dimension")
            padded[index, : features.shape[0]] = features.float()
        else:
            padded[index, : features.shape[0]] = features.long()

        encoded = [token_id + 1 for token_id in vocab.encode(example.normalized_text)]
        targets.extend(encoded)
        target_lengths.append(len(encoded))

    return CTCBatch(
        utterance_ids=[example.utterance_id for example in examples],
        features=padded,
        feature_lengths=feature_lengths,
        targets=torch.tensor(targets, dtype=torch.long),
        target_lengths=torch.tensor(target_lengths, dtype=torch.long),
        texts=[example.text for example in examples],
        normalized_texts=[example.normalized_text for example in examples],
    )


def collate_seq2seq(examples: list[Seq2SeqExample]) -> Seq2SeqBatch:
    if not examples:
        raise ValueError("examples must not be empty")

    feature_lengths = torch.tensor([example.features.shape[0] for example in examples], dtype=torch.long)
    feature_rank = examples[0].features.ndim
    max_frames = int(feature_lengths.max().item())
    if feature_rank == 1:
        features = torch.zeros(len(examples), max_frames, dtype=torch.long)
    elif feature_rank == 2:
        feature_dim = examples[0].features.shape[-1]
        features = torch.zeros(len(examples), max_frames, feature_dim, dtype=torch.float32)
    else:
        raise ValueError(f"features must be 1-D or 2-D, got shape {tuple(examples[0].features.shape)}")

    decoder_sequences = []
    label_sequences = []
    for index, example in enumerate(examples):
        if example.features.ndim != feature_rank:
            raise ValueError("all examples in a batch must have the same feature rank")
        if feature_rank == 2:
            if example.features.shape[-1] != feature_dim:
                raise ValueError("all examples in a batch must have the same feature dimension")
            features[index, : example.features.shape[0]] = example.features.float()
        else:
            features[index, : example.features.shape[0]] = example.features.long()

        token_ids = encode_seq2seq_text(example.normalized_text)
        decoder_sequences.append([1, *token_ids])
        label_sequences.append([*token_ids, 2])

    max_tokens = max(len(sequence) for sequence in decoder_sequences)
    decoder_inputs = torch.zeros(len(examples), max_tokens, dtype=torch.long)
    labels = torch.zeros(len(examples), max_tokens, dtype=torch.long)
    for index, (decoder_sequence, label_sequence) in enumerate(zip(decoder_sequences, label_sequences)):
        decoder_inputs[index, : len(decoder_sequence)] = torch.tensor(decoder_sequence, dtype=torch.long)
        labels[index, : len(label_sequence)] = torch.tensor(label_sequence, dtype=torch.long)

    return Seq2SeqBatch(
        utterance_ids=[example.utterance_id for example in examples],
        features=features,
        feature_lengths=feature_lengths,
        decoder_inputs=decoder_inputs,
        labels=labels,
        texts=[example.text for example in examples],
        normalized_texts=[example.normalized_text for example in examples],
    )


def encode_seq2seq_text(text: str) -> list[int]:
    vocab = Vocabulary.from_symbols(CHAR_SYMBOLS)
    return [token_id + 3 for token_id in vocab.encode(text)]


def decode_seq2seq_tokens(token_ids: list[int]) -> str:
    symbols = []
    for token_id in token_ids:
        if token_id in (0, 1):
            continue
        if token_id == 2:
            break
        symbol_index = token_id - 3
        if symbol_index < 0 or symbol_index >= len(CHAR_SYMBOLS):
            continue
        symbols.append(CHAR_SYMBOLS[symbol_index])
    return "".join(symbols)
