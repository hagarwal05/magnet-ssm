"""The S4D model: kernel correctness, causality, size control, trainability."""
import numpy as np
import pytest
import torch

from src.models import SSMConfig, SSMNet, build_model, count_parameters
from src.models.ssm import S4DKernel


def _recurrent_kernel(k: S4DKernel, L: int) -> torch.Tensor:
    """Impulse response of the ZOH-discretized diagonal SSM computed by complex recurrence:
    x_{t+1} = exp(dt A) x_t + B' u_t, y_t = 2 Re(C x_t) with B' = (exp(dt A) - 1)/A."""
    dt = torch.exp(k.log_dt).double().unsqueeze(-1)
    A = (-torch.exp(k.log_A_real) + 1j * k.A_imag).to(torch.complex128)
    C = (k.C_re + 1j * k.C_im).to(torch.complex128)
    dA = torch.exp(dt * A)
    dB = (dA - 1) / A
    x = dB.clone()          # state after an impulse at t = 0
    out = []
    for _ in range(L):
        out.append(2 * (C * x).sum(-1).real)
        x = dA * x
    return torch.stack(out, -1)


def test_kernel_matches_recurrence():
    torch.manual_seed(0)
    k = S4DKernel(channels=3, d_state=8, dt_min=0.01, dt_max=0.5)
    with torch.no_grad():
        np.testing.assert_allclose(k(40).double().numpy(), _recurrent_kernel(k, 40).numpy(), rtol=1e-4, atol=1e-5)


def test_causal():
    """Changing the last window step must not change features of earlier steps."""
    torch.manual_seed(0)
    net = SSMNet(SSMConfig(d_model=4, n_layers=2)).eval()
    x = torch.randn(2, 80, 2)
    h1 = net.encoder(x)
    x2 = x.clone()
    x2[:, -1] += 5.0
    h2 = net.encoder(x2)
    for blk in net.blocks:
        h1, h2 = blk(h1), blk(h2)
    torch.testing.assert_close(h1[:, :-1], h2[:, :-1])
    assert not torch.allclose(h1[:, -1], h2[:, -1])


def test_shapes_and_registry():
    net = build_model("ssm", {"d_model": 6, "n_layers": 1})
    out = net(torch.randn(5, 80, 1), torch.randn(5, 80, 1), torch.randn(5, 1), torch.randn(5, 1))
    assert out.shape == (5, 1)
    with pytest.raises(ValueError):
        build_model("official_lstm", {"d_model": 3})


def test_param_count_scales_with_config():
    counts = [count_parameters(SSMNet(SSMConfig(d_model=w, n_layers=l))) for w, l in [(4, 1), (8, 1), (8, 2), (16, 2)]]
    assert counts == sorted(counts) and len(set(counts)) == 4
    # the small end sits below the 771-parameter official baseline
    assert counts[0] < 771


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_overfits_one_batch(device):
    torch.manual_seed(0)
    net = SSMNet(SSMConfig(d_model=8, n_layers=2)).to(device)
    sB, sH = torch.randn(64, 80, 1, device=device), torch.randn(64, 80, 1, device=device)
    bN, T = torch.randn(64, 1, device=device), torch.randn(64, 1, device=device)
    y = (0.5 * sH[:, -1] + bN - 0.3 * sB[:, -5]).detach()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    first = None
    for _ in range(300):
        opt.zero_grad()
        loss = ((net(sB, sH, bN, T) - y) ** 2).mean()
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.05 * first
