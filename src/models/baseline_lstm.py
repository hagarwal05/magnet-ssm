"""Faithful re-implementations of the two upstream dual-LSTM baselines.

Attribute names (``lstm_B``, ``lstm_H``, ``projector``) and layer order match the upstream
``class Net`` exactly, so upstream ``.sd`` state dicts load with ``strict=True``.

* ``OfficialLSTM``: ``Test/EvaluationExample.ipynb`` and ``Pretest/PreEvaluationExample.ipynb``.
  Two LSTM(1, 6), Tanh projector 14 -> 14 -> 8 -> 1. 771 parameters.
  Matches the ten ``Pretest/Pretest_Models/Model_LSTM_<M>.sd`` files.
* ``TutorialLSTM``: tutorial-1 training/testing and tutorial-3 evaluation notebooks.
  Two LSTM(1, 12), ReLU projector 26 -> 26 -> 8 -> 4 -> 1. 2,399 parameters.
  Matches ``tutorial-1/Model_LSTM_{Princeton,New}.sd`` and ``tutorial-3/Model_LSTM.sd``.

Both share the interface of every model in this package:
``forward(seq_B[N,W,1], seq_H[N,W,1], B_next[N,1], T[N,1]) -> H_next[N,1]`` (normalized units).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _DualLSTM(nn.Module):
    def __init__(self, hidden: int, projector: nn.Sequential):
        super().__init__()
        self.lstm_B = nn.LSTM(1, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_H = nn.LSTM(1, hidden, num_layers=1, batch_first=True, bidirectional=False)
        self.projector = projector

    def forward(self, seq_B: torch.Tensor, seq_H: torch.Tensor, B_next: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        x_B, _ = self.lstm_B(seq_B.float())
        x_H, _ = self.lstm_H(seq_H.float())
        feats = torch.cat((B_next.float(), T.float(), x_B[:, -1, :], x_H[:, -1, :]), dim=1)
        return self.projector(feats)


class OfficialLSTM(_DualLSTM):
    """771-parameter evaluator baseline (hidden 6, Tanh, 3 linear layers)."""

    def __init__(self):
        h = 6
        super().__init__(h, nn.Sequential(
            nn.Linear(h * 2 + 2, h * 2 + 2),
            nn.Tanh(),
            nn.Linear(h * 2 + 2, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        ))


class TutorialLSTM(_DualLSTM):
    """2,399-parameter tutorial baseline (hidden 12, ReLU, 4 linear layers)."""

    def __init__(self):
        h = 12
        super().__init__(h, nn.Sequential(
            nn.Linear(h * 2 + 2, h * 2 + 2),
            nn.ReLU(),
            nn.Linear(h * 2 + 2, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        ))
