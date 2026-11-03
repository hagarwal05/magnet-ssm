"""Model registry. Every model maps
``(seq_B[N,W,1], seq_H[N,W,1], B_next[N,1], T[N,1]) -> H_next[N,1]`` in normalized units."""
from __future__ import annotations

import torch.nn as nn

from .baseline_lstm import OfficialLSTM, TutorialLSTM
from .ssm import SSMConfig, SSMNet

MODELS = {
    "official_lstm": OfficialLSTM,
    "tutorial_lstm": TutorialLSTM,
    "ssm": SSMNet,
}


def build_model(name: str, config: dict | None = None) -> nn.Module:
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    if name == "ssm":
        return SSMNet(SSMConfig(**(config or {})))
    if config:
        raise ValueError(f"{name} is a fixed reference architecture and takes no config")
    return MODELS[name]()


def model_config(model: nn.Module) -> dict:
    return model.cfg.to_dict() if isinstance(model, SSMNet) else {}


def count_parameters(model: nn.Module) -> int:
    """Trainable parameter count, the challenge's definition."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["MODELS", "build_model", "model_config", "count_parameters",
           "OfficialLSTM", "TutorialLSTM", "SSMNet", "SSMConfig"]
