"""Eager end-to-end test for the CONSTANT_SNR gradient-SNR collector (csnr_meter.py).

Constructs a tiny Linear model with a KNOWN per-timestep gradient SNR: samples at bin b
get target Wx + delta_b*d_b + noise, so the output-gradient's per-bin signal is delta_b*d_b
(||.||^2 = delta_b^2) over fixed noise -- SNR_b ~ delta_b^2, set to DECREASE across bins.
Then runs the real hook -> sketch -> deconvolution -> guard -> EMA -> curve path and checks
the collector recovers the ordering. This is the reviewable reference the graph-native
kernel projection buffer must match; run with the repo root on PYTHONPATH:

    python -m modules.util.optimizer.concord.tests.test_csnr_meter
"""
import torch
import torch.nn as nn

from modules.util.optimizer.csnr_meter import CSNRCollector

D, NB, B, T, STEPS = 128, 8, 64, 1000, 50


def _spearman(a, b):
    ra = a.argsort().argsort().float(); rb = b.argsort().argsort().float()
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / (ra.norm() * rb.norm() + 1e-30))


def _one_window(coll, model, x0, Wx, d, delta, sigma, gen):
    edges = torch.linspace(0, T, NB + 1)
    coll.arm()
    for _ in range(STEPS):
        comp = torch.randint(0, NB, (B,), generator=gen)
        ts = (edges[comp] + torch.rand(B, generator=gen) * (T / NB)).long().clamp(max=T - 1)
        xin = x0.unsqueeze(0).repeat(B, 1)
        t = Wx.unsqueeze(0) + delta[comp, None] * d[comp] + torch.randn(B, D, generator=gen) * sigma
        loss = ((model(xin) - t) ** 2).sum(1).mean()
        model.zero_grad(); loss.backward()
        coll.observe(ts)
    return coll.emit()


def _setup(seed):
    torch.manual_seed(seed)
    model = nn.Linear(D, D, bias=False)
    x0 = torch.randn(D)
    gen = torch.Generator().manual_seed(seed + 1)
    d = torch.randn(NB, D, generator=gen); d /= d.norm(dim=1, keepdim=True)
    delta = torch.linspace(1.0, 0.3, NB); sigma = 0.5
    Wx = model(x0).detach()
    snr_true = (delta ** 2) / (sigma ** 2)
    edges = torch.linspace(0, T, NB + 1)
    mids = ((edges[:-1] + edges[1:]) / 2).long().clamp(max=T - 1)
    return model, x0, Wx, d, delta, sigma, snr_true, mids, gen


def test_single_window_recovers_ordering():
    """A kept window recovers the SNR ordering; degenerate windows are rejected (None)."""
    kept = []
    for seed in range(10):
        model, x0, Wx, d, delta, sigma, snr_true, mids, gen = _setup(seed)
        coll = CSNRCollector([model], T, window=STEPS, K=64, n_bins=NB, ema=0.0, prior_w=0.0)
        out = _one_window(coll, model, x0, Wx, d, delta, sigma, gen)
        if out is not None:
            kept.append(_spearman(snr_true, out[0][mids]))
    assert len(kept) >= 7, f"too many windows rejected: kept {len(kept)}/10"
    assert sum(kept) / len(kept) > 0.75, f"mean ordering too low: {sum(kept)/len(kept):.3f}"


def test_ema_is_robust_to_bad_windows():
    """EMA across windows holds a correctly-ordered curve even when single windows are noisy."""
    model, x0, Wx, d, delta, sigma, snr_true, mids, gen = _setup(0)
    coll = CSNRCollector([model], T, window=STEPS, K=64, n_bins=NB, ema=0.7, prior_w=0.0)
    last = None
    for _ in range(8):
        out = _one_window(coll, model, x0, Wx, d, delta, sigma, gen)
        if out is not None:
            last = _spearman(snr_true, out[0][mids])
    assert last is not None and last > 0.9, f"EMA curve not converged: {last}"


if __name__ == "__main__":
    test_single_window_recovers_ordering()
    test_ema_is_robust_to_bad_windows()
    print("test_csnr_meter: PASS")
