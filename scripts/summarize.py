"""Summarize ASR experiment metrics into a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence


def collect_metrics(outputs_dir: Path) -> list[dict]:
    rows = []
    for metrics_path in sorted(outputs_dir.glob("*/metrics.json")):
        rows.append({"experiment": metrics_path.parent.name, "split": "dev", **read_json(metrics_path)})
    for metrics_path in sorted(outputs_dir.glob("*/eval/*.json")):
        rows.append(
            {
                "experiment": metrics_path.parents[1].name,
                "split": metrics_path.stem,
                **read_json(metrics_path),
            }
        )
    return rows


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", "split", "wer", "cer", "num_utts", "epoch", "train_loss"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics.")
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("results/summary.csv"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = collect_metrics(args.outputs_dir)
    write_summary(rows, args.output)
    print(f"[summary] wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
