"""Discrete speech-unit utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence


def deduplicate_units(units: Sequence[int]) -> tuple[list[int], list[int]]:
    if not units:
        return [], []

    deduped = [int(units[0])]
    lengths = [1]
    for unit in units[1:]:
        unit = int(unit)
        if unit == deduped[-1]:
            lengths[-1] += 1
        else:
            deduped.append(unit)
            lengths.append(1)
    return deduped, lengths


def duration_buckets(run_lengths: Sequence[int]) -> list[int]:
    buckets = []
    for run_length in run_lengths:
        if run_length < 1:
            raise ValueError(f"run_length must be positive, got {run_length}")
        buckets.append(min(7, int(math.floor(math.log2(run_length)))))
    return buckets


def token_rate(*, num_tokens: int, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    return num_tokens / duration_seconds


def fixed_code_bitrate(
    *,
    token_rate_hz: float,
    codebook_size: int,
    duration_bits: int = 0,
) -> float:
    if codebook_size < 2:
        raise ValueError("codebook_size must be at least 2")
    if token_rate_hz < 0:
        raise ValueError("token_rate_hz must be non-negative")
    if duration_bits < 0:
        raise ValueError("duration_bits must be non-negative")
    bits_per_token = math.ceil(math.log2(codebook_size)) + duration_bits
    return token_rate_hz * bits_per_token
