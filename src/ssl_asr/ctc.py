"""CTC decoding helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def greedy_ctc_decode(
    token_ids: Sequence[int],
    id_to_token: Mapping[int, str],
    *,
    blank_id: int,
) -> str:
    """Collapse repeated CTC predictions and remove blank tokens."""
    pieces = []
    previous_id: int | None = None
    for token_id in token_ids:
        if token_id == blank_id:
            previous_id = blank_id
            continue
        if token_id == previous_id:
            continue
        if token_id not in id_to_token:
            raise KeyError(f"Unknown token id: {token_id}")
        pieces.append(id_to_token[token_id])
        previous_id = token_id
    return "".join(pieces)
