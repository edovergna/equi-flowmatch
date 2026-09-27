"""Shared CUDA training utilities for the command-line experiments."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def base_parser(description: str, default_epochs: int, default_lr: float) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--epochs", type=int, default=default_epochs)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=default_lr)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Limit batches per epoch for a smoke test; omit for full training.",
    )
    return parser


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required. Request a GPU node on Snellius and install "
            "a CUDA-enabled PyTorch build."
        )
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}", flush=True)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resume_if_requested(model, optimizer, path: Path | None) -> int:
    if path is None:
        return 0
    checkpoint = torch.load(path, map_location="cuda", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    if isinstance(checkpoint, dict) and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0


def save_checkpoint(path: Path, model, optimizer, epoch: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "arguments": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def report(epoch: int, iteration: int, total: int, loss: torch.Tensor) -> None:
    print(
        f"epoch {epoch + 1}, batch {iteration + 1}/{total}, loss {loss.item():.6g}",
        flush=True,
    )
