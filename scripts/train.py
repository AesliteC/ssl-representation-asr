"""Train ASR downstream models from YAML experiment configs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ssl_asr.ctc import greedy_ctc_decode
from ssl_asr.data.asr_dataset import ASRDataset
from ssl_asr.data.training import collate_ctc, collate_seq2seq, decode_seq2seq_tokens
from ssl_asr.metrics import char_error_rate, word_error_rate
from ssl_asr.models.ctc import BiLSTMCTC
from ssl_asr.models.transformer import TransformerASR
from ssl_asr.text import CHAR_SYMBOLS


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    manifest_path: Path,
    *,
    config: dict,
    root: Path,
    limit: int | None,
    shuffle: bool,
) -> DataLoader:
    dataset = ASRDataset(
        manifest_path,
        representation=config["representation"],
        root=root,
        limit=limit,
    )
    model_type = config["model"]["type"]
    collate_fn = collate_ctc if model_type == "ctc" else collate_seq2seq
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"].get("num_workers", 0)),
        collate_fn=collate_fn,
    )


def build_model(config: dict) -> nn.Module:
    representation = config["representation"]
    model_config = config["model"]
    model_type = model_config["type"]
    is_discrete = representation["kind"] == "discrete"
    codebook_size = int(representation.get("codebook_size", 0))
    if is_discrete and representation.get("use_duration"):
        codebook_size *= 8

    if model_type == "ctc":
        return BiLSTMCTC(
            vocab_size=len(CHAR_SYMBOLS) + 1,
            input_dim=None if is_discrete else int(representation["input_dim"]),
            codebook_size=codebook_size if is_discrete else None,
            hidden_size=int(model_config.get("hidden_size", 256)),
            projection_dim=int(model_config.get("projection_dim", 256)),
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.2)),
        )
    if model_type == "transformer":
        return TransformerASR(
            vocab_size=len(CHAR_SYMBOLS) + 3,
            input_dim=None if is_discrete else int(representation["input_dim"]),
            codebook_size=codebook_size if is_discrete else None,
            d_model=int(model_config.get("d_model", 256)),
            num_encoder_layers=int(model_config.get("num_encoder_layers", 4)),
            num_decoder_layers=int(model_config.get("num_decoder_layers", 3)),
            nhead=int(model_config.get("nhead", 4)),
            dim_feedforward=int(model_config.get("dim_feedforward", 1024)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def train_ctc(model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader, config: dict, device: torch.device) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best = {"wer": float("inf"), "cer": float("inf"), "epoch": 0}
    patience = int(config["training"].get("patience", 8))
    stale_epochs = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            features = batch.features.to(device)
            feature_lengths = batch.feature_lengths.to(device)
            targets = batch.targets.to(device)
            target_lengths = batch.target_lengths.to(device)
            logits, output_lengths = model(features, feature_lengths)
            loss = criterion(logits.log_softmax(dim=-1), targets, output_lengths.cpu(), target_lengths.cpu())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 5.0)))
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        metrics = evaluate_ctc(model, dev_loader, device)
        metrics["train_loss"] = total_loss / max(1, total_batches)
        metrics["epoch"] = epoch
        print(f"[train] epoch={epoch} loss={metrics['train_loss']:.4f} dev_wer={metrics['wer']:.4f} dev_cer={metrics['cer']:.4f}")
        if metrics["wer"] < best["wer"]:
            best = metrics
            stale_epochs = 0
            save_checkpoint(model, config, best, Path(config["output_dir"]) / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return best


def train_transformer(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    best = {"wer": float("inf"), "cer": float("inf"), "epoch": 0}
    patience = int(config["training"].get("patience", 8))
    stale_epochs = 0

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch.features.to(device),
                batch.feature_lengths.to(device),
                batch.decoder_inputs.to(device),
            )
            labels = batch.labels.to(device)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("grad_clip", 5.0)))
            optimizer.step()
            total_loss += float(loss.item())
            total_batches += 1

        metrics = evaluate_transformer(model, dev_loader, device)
        metrics["train_loss"] = total_loss / max(1, total_batches)
        metrics["epoch"] = epoch
        print(f"[train] epoch={epoch} loss={metrics['train_loss']:.4f} dev_wer={metrics['wer']:.4f} dev_cer={metrics['cer']:.4f}")
        if metrics["wer"] < best["wer"]:
            best = metrics
            stale_epochs = 0
            save_checkpoint(model, config, best, Path(config["output_dir"]) / "best.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    return best


def evaluate_ctc(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    id_to_token = {idx + 1: token for idx, token in enumerate(CHAR_SYMBOLS)}
    predictions = []
    references = []
    with torch.inference_mode():
        for batch in loader:
            logits, _ = model(batch.features.to(device), batch.feature_lengths.to(device))
            token_ids = logits.argmax(dim=-1).cpu()
            for batch_index, length in enumerate(batch.feature_lengths.tolist()):
                predicted = greedy_ctc_decode(token_ids[:length, batch_index].tolist(), id_to_token, blank_id=0)
                predictions.append(predicted)
                references.append(batch.normalized_texts[batch_index])
    return aggregate_metrics(references, predictions)


def evaluate_transformer(model: nn.Module, loader: DataLoader, device: torch.device, *, max_tokens: int = 256) -> dict:
    model.eval()
    predictions = []
    references = []
    with torch.inference_mode():
        for batch in loader:
            features = batch.features.to(device)
            feature_lengths = batch.feature_lengths.to(device)
            generated = torch.ones(features.size(0), 1, dtype=torch.long, device=device)
            for _ in range(max_tokens):
                logits = model(features, feature_lengths, generated)
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.eq(2).all():
                    break
            for token_ids, reference in zip(generated.cpu().tolist(), batch.normalized_texts):
                predictions.append(decode_seq2seq_tokens(token_ids))
                references.append(reference)
    return aggregate_metrics(references, predictions)


def aggregate_metrics(references: list[str], predictions: list[str]) -> dict:
    if not references:
        return {"wer": 0.0, "cer": 0.0, "num_utts": 0}
    wers = [word_error_rate(reference, prediction) for reference, prediction in zip(references, predictions)]
    cers = [char_error_rate(reference, prediction) for reference, prediction in zip(references, predictions)]
    return {
        "wer": float(sum(wers) / len(wers)),
        "cer": float(sum(cers) / len(cers)),
        "num_utts": len(references),
    }


def save_checkpoint(model: nn.Module, config: dict, metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": config,
            "metrics": metrics,
            "state_dict": model.state_dict(),
        },
        path,
    )
    (path.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_training(config_path: Path, *, root: Path, device_name: str, limit_train: int | None, limit_dev: int | None) -> dict:
    config = load_config(config_path)
    config["output_dir"] = str(root / config["output_dir"])
    set_seed(int(config.get("seed", 42)))
    device = torch.device(device_name)
    model = build_model(config).to(device)
    train_loader = build_dataloader(
        root / config["data"]["train_manifest"],
        config=config,
        root=root,
        limit=limit_train,
        shuffle=True,
    )
    dev_loader = build_dataloader(
        root / config["data"]["dev_manifest"],
        config=config,
        root=root,
        limit=limit_dev,
        shuffle=False,
    )
    if config["model"]["type"] == "ctc":
        best = train_ctc(model, train_loader, dev_loader, config, device)
    else:
        best = train_transformer(model, train_loader, dev_loader, config, device)
    (Path(config["output_dir"]) / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return best


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one ASR experiment config.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args.config, root=args.root.resolve(), device_name=args.device, limit_train=args.limit_train, limit_dev=args.limit_dev)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
