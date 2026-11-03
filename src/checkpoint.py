"""Checkpoints that carry their own normalization, so weights and normalization can never be
mixed up. Format: a plain dict saved with ``torch.save`` and read with
``weights_only=True`` (tensors, dicts and primitives only)."""
from __future__ import annotations

from pathlib import Path

import torch

from .models import build_model, count_parameters, model_config
from .normalize import NormParams

FORMAT_VERSION = 1


def save_checkpoint(path: str | Path, model: torch.nn.Module, model_name: str, norm: NormParams,
                    meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "model_name": model_name,
        "config": model_config(model),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "norm": norm.to_dict(),
        "n_params": count_parameters(model),
        "meta": meta or {},
    }, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu"):
    """Returns (model, norm, checkpoint_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict) or "state_dict" not in ck or "norm" not in ck:
        raise ValueError(f"{path} is not a checkpoint written by save_checkpoint "
                         "(for upstream .sd files use --reference or --weights with --norm)")
    model = build_model(ck["model_name"], ck.get("config") or None)
    model.load_state_dict(ck["state_dict"], strict=True)
    return model.to(device), NormParams.from_dict(ck["norm"]), ck


def load_state_dict_file(path: str | Path) -> dict:
    """Load an upstream ``.sd`` (a bare state dict) safely."""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(sd).__name__}")
    return sd
