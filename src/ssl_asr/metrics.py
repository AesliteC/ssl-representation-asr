"""ASR edit-distance metrics."""

from __future__ import annotations

from collections.abc import Sequence


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein edit distance."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_item in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_item in enumerate(hypothesis, start=1):
            deletion = previous[hyp_index] + 1
            insertion = current[hyp_index - 1] + 1
            substitution = previous[hyp_index - 1] + (ref_item != hyp_item)
            current.append(min(deletion, insertion, substitution))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return edit_distance(reference_words, hypothesis_words) / len(reference_words)


def char_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = tuple(reference.replace(" ", ""))
    hypothesis_chars = tuple(hypothesis.replace(" ", ""))
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    return edit_distance(reference_chars, hypothesis_chars) / len(reference_chars)
