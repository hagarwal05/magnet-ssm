"""Diagonal state-space sequence model (S4D) for next-step H prediction.

Same interface as the baselines:
``forward(seq_B[N,W,1], seq_H[N,W,1], B_next[N,1], T[N,1]) -> H_next[N,1]`` (normalized).

Architecture
  * Encoder: the window is a 2-channel sequence (B_t, H_t), lifted to ``d_model`` channels
    by a pointwise linear layer.
  * ``n_layers`` residual S4D blocks (pre-LayerNorm -> S4D convolution + D skip -> GELU ->
    GLU output mixing). Each channel holds ``d_state // 2`` complex-conjugate modes with
    S4D-Lin initialization (A = -1/2 + i*pi*n) and a learned step size.
  * Head: the last-step features, concatenated with [B_next, T] the way the baseline
    concatenates its LSTM states, go through a small Tanh MLP to the next H.

The SSM kernel follows the minimal S4D reference (state-spaces/s4, ``models/s4/s4d.py``,
Apache-2.0) with ZOH discretization, but is written in real arithmetic and applied as a
causal depthwise convolution. That runs on every PyTorch backend (CPU, CUDA, MPS) without
complex-tensor or FFT support. Window lengths are short (80), so direct convolution is cheap.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SSMConfig:
    d_model: int = 8         # channels per layer (width)
    d_state: int = 16        # state size per channel (even; d_state // 2 complex modes)
    n_layers: int = 2        # number of S4D blocks (depth)
    head_hidden: int = 8     # hidden width of the output MLP (0 = linear head)
    dt_min: float = 0.01     # step-size init range, in units of one 16 MHz sample
    dt_max: float = 0.1
    residual_head: bool = False  # predict H_next - H_last instead of H_next
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class S4DKernel(nn.Module):
    """Generates the length-L convolution kernel of a diagonal SSM, one per channel."""

    def __init__(self, channels: int, d_state: int, dt_min: float, dt_max: float):
        super().__init__()
        if d_state % 2:
            raise ValueError(f"d_state must be even, got {d_state}")
        n = d_state // 2
        log_dt = torch.rand(channels) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.C_re = nn.Parameter(torch.randn(channels, n) * (0.5 ** 0.5))
        self.C_im = nn.Parameter(torch.randn(channels, n) * (0.5 ** 0.5))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(channels, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n, dtype=torch.float32).repeat(channels, 1))

    def forward(self, L: int) -> torch.Tensor:
        """Returns K of shape (channels, L): K[h, l] = 2 Re(sum_n C'_hn exp(dt A_hn l))."""
        dt = torch.exp(self.log_dt).unsqueeze(-1)            # (H, 1)
        a_re = -torch.exp(self.log_A_real)                    # (H, n), < 0 for stability
        a_im = self.A_imag
        # ZOH: C' = C * (exp(dt A) - 1) / A, in real arithmetic
        e = torch.exp(dt * a_re)
        p = e * torch.cos(dt * a_im) - 1.0                    # Re(exp(dt A) - 1)
        q = e * torch.sin(dt * a_im)                          # Im(exp(dt A) - 1)
        denom = a_re ** 2 + a_im ** 2
        r_re = (p * a_re + q * a_im) / denom
        r_im = (q * a_re - p * a_im) / denom
        c_re = self.C_re * r_re - self.C_im * r_im
        c_im = self.C_re * r_im + self.C_im * r_re
        # Vandermonde over time steps
        l = torch.arange(L, device=dt.device, dtype=dt.dtype)
        decay = torch.exp((dt * a_re).unsqueeze(-1) * l)     # (H, n, L)
        phase = (dt * a_im).unsqueeze(-1) * l
        K = 2.0 * (c_re.unsqueeze(-1) * decay * torch.cos(phase)
                   - c_im.unsqueeze(-1) * decay * torch.sin(phase)).sum(dim=1)
        return K


class S4DBlock(nn.Module):
    def __init__(self, cfg: SSMConfig):
        super().__init__()
        h = cfg.d_model
        self.norm = nn.LayerNorm(h)
        self.kernel = S4DKernel(h, cfg.d_state, cfg.dt_min, cfg.dt_max)
        self.D = nn.Parameter(torch.randn(h))
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()
        self.out = nn.Linear(h, 2 * h)  # GLU mixing across channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, L, H) -> (N, L, H)."""
        N, L, H = x.shape
        u = self.norm(x).transpose(1, 2)                      # (N, H, L)
        k = self.kernel(L)                                    # (H, L)
        # causal depthwise convolution: y[t] = sum_{j<=t} k[j] u[t - j]
        y = F.conv1d(F.pad(u, (L - 1, 0)), k.flip(-1).unsqueeze(1), groups=H)
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(F.gelu(y)).transpose(1, 2)           # (N, L, H)
        return x + F.glu(self.out(y), dim=-1)


class SSMNet(nn.Module):
    def __init__(self, cfg: SSMConfig | None = None, **overrides):
        super().__init__()
        cfg = cfg or SSMConfig()
        if overrides:
            cfg = SSMConfig(**{**cfg.to_dict(), **overrides})
        self.cfg = cfg
        h = cfg.d_model
        self.encoder = nn.Linear(2, h)
        self.blocks = nn.ModuleList(S4DBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(h)
        if cfg.head_hidden > 0:
            self.head = nn.Sequential(nn.Linear(h + 2, cfg.head_hidden), nn.Tanh(),
                                      nn.Linear(cfg.head_hidden, 1))
        else:
            self.head = nn.Linear(h + 2, 1)

    def forward(self, seq_B: torch.Tensor, seq_H: torch.Tensor, B_next: torch.Tensor,
                T: torch.Tensor) -> torch.Tensor:
        x = self.encoder(torch.cat((seq_B.float(), seq_H.float()), dim=-1))   # (N, L, H)
        for blk in self.blocks:
            x = blk(x)
        feat = self.final_norm(x[:, -1, :])
        out = self.head(torch.cat((B_next.float(), T.float(), feat), dim=1))
        if self.cfg.residual_head:
            out = out + seq_H[:, -1, :].float()
        return out
