"""CTC acoustic models."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMCTC(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        input_dim: int | None = None,
        codebook_size: int | None = None,
        hidden_size: int = 256,
        projection_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.use_embedding = codebook_size is not None
        if self.use_embedding:
            self.embedding = nn.Embedding(codebook_size, projection_dim)
            lstm_input_dim = projection_dim
            self.input_norm = None
            self.input_projection = None
        else:
            if input_dim is None:
                raise ValueError("input_dim is required for continuous features")
            self.embedding = None
            self.input_norm = nn.LayerNorm(input_dim)
            self.input_projection = nn.Linear(input_dim, projection_dim)
            lstm_input_dim = projection_dim

        self.encoder = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_embedding:
            encoded = self.embedding(features.long())
        else:
            encoded = self.input_projection(self.input_norm(features.float()))

        packed = nn.utils.rnn.pack_padded_sequence(
            encoded,
            feature_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.classifier(output).transpose(0, 1)
        return logits, feature_lengths
