"""Manifest helpers for LibriSpeech-style transcript trees."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from ssl_asr.text import normalize_text


@dataclass(frozen=True)
class ManifestRecord:
    utterance_id: str
    audio_path: Path
    text: str
    normalized_text: str
    speaker_id: str
    chapter_id: str
    split: str

    def to_json(self, *, audio_root: Path | None = None) -> dict[str, str]:
        if audio_root is None:
            audio_path = self.audio_path.resolve()
        else:
            audio_path = self.audio_path.resolve().relative_to(audio_root.resolve())

        row = asdict(self)
        row["audio_filepath"] = audio_path.as_posix()
        row.pop("audio_path")
        return row


def iter_transcript_tree(split_dir: Path, *, split: str | None = None) -> Iterator[ManifestRecord]:
    """Yield records from any tree containing LibriSpeech-style .trans.txt files."""
    split_dir = split_dir.resolve()
    split_name = split or split_dir.name
    for transcript_path in sorted(split_dir.rglob("*.trans.txt")):
        chapter_dir = transcript_path.parent
        speaker_id = chapter_dir.parent.name
        chapter_id = chapter_dir.name
        for utterance_id, text in read_transcript(transcript_path):
            audio_path = chapter_dir / f"{utterance_id}.flac"
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            yield ManifestRecord(
                utterance_id=utterance_id,
                audio_path=audio_path,
                text=text,
                normalized_text=normalize_text(text),
                speaker_id=speaker_id,
                chapter_id=chapter_id,
                split=split_name,
            )


def read_transcript(transcript_path: Path) -> Iterator[tuple[str, str]]:
    for line_number, line in enumerate(transcript_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            utterance_id, text = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"Invalid transcript line {transcript_path}:{line_number}") from exc
        yield utterance_id, text


def write_manifest(
    records: Iterable[ManifestRecord],
    output_path: Path,
    *,
    audio_root: Path | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_json(audio_root=audio_root), ensure_ascii=False))
            handle.write("\n")
