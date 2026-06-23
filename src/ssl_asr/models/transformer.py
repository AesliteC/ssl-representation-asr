"""Transformer encoder-decoder ASR model."""

from __future__ import annotations

import torch
from torch import nn


class TransformerASR(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        input_dim: int | None = None,
        codebook_size: int | None = None,
        d_model: int = 256,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.use_unit_embedding = codebook_size is not None
        if self.use_unit_embedding:
            self.input_embedding = nn.Embedding(codebook_size, d_model)
            self.input_projection = None
        else:
            if input_dim is None:
                raise ValueError("input_dim is required for continuous inputs")
            self.input_embedding = None
            self.input_projection = nn.Linear(input_dim, d_model)
        self.subsample = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: torch.Tensor,
        decoder_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_unit_embedding:
            src = self.input_embedding(features.long())
        else:
            src = self.input_projection(features.float())
        src = self.subsample(src.transpose(1, 2)).transpose(1, 2)
        src_lengths = (feature_lengths + 1) // 2
        src_padding_mask = lengths_to_padding_mask(src_lengths, src.size(1))

        tgt = self.token_embedding(decoder_tokens)
        tgt_mask = torch.triu(
            torch.ones(tgt.size(1), tgt.size(1), dtype=torch.bool, device=tgt.device),
            diagonal=1,
        )
        tgt_padding_mask = decoder_tokens.eq(self.pad_id)
        output = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_padding_mask.to(src.device),
            tgt_key_padding_mask=tgt_padding_mask,
            tgt_mask=tgt_mask,
        )
        return self.classifier(output)


def lengths_to_padding_mask(lengths: torch.Tensor, max_length: int) -> torch.Tensor:
    positions = torch.arange(max_length, device=lengths.device).unsqueeze(0)
    return positions >= lengths.unsqueeze(1)
