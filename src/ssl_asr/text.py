"""Text normalization and character vocabulary helpers."""

from __future__ import annotations

from dataclasses import dataclass


CHAR_SYMBOLS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ' ")
_ALLOWED_CHARS = set(CHAR_SYMBOLS)


def normalize_text(text: str) -> str:
    """Normalize transcript text for character-level ASR."""
    normalized = []
    for char in text.upper():
        if char in _ALLOWED_CHARS:
            normalized.append(char)
        elif char.isspace():
            normalized.append(" ")
    return " ".join("".join(normalized).split())


@dataclass(frozen=True)
class Vocabulary:
    symbols: tuple[str, ...]
    token_to_id: dict[str, int]

    @classmethod
    def from_symbols(cls, symbols: tuple[str, ...]) -> "Vocabulary":
        if len(set(symbols)) != len(symbols):
            raise ValueError("Vocabulary symbols must be unique")
        return cls(symbols=symbols, token_to_id={token: idx for idx, token in enumerate(symbols)})

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char not in self.token_to_id:
                raise ValueError(f"Unsupported character: {char!r}")
            ids.append(self.token_to_id[char])
        return ids

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> str:
        pieces = []
        for token_id in token_ids:
            try:
                pieces.append(self.symbols[token_id])
            except IndexError as exc:
                raise KeyError(f"Unknown token id: {token_id}") from exc
        return "".join(pieces)
