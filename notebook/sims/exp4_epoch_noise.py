"""Does epoch-structured noise (sums to ~0 each epoch, like shuffled
minibatch sampling) make signal/noise separable where i.i.d. white noise
(a free Brownian walk) did not?

Per step: g = mu (drift, persists across epochs) + noise_t, where noise_t
over each epoch of E steps is mean-subtracted -> sums to 0 (bridge, not walk).
Control: i.i.d. white noise (does NOT cancel -> free walk).

Measure coh_persist = |EMA(d_sv)|^2 / EMA(d_sv^2), EMA window ~ a few epochs.
Want: epoch-noise -> signal high, PURE NOISE low (clean separation);
white-noise control -> noise false-positives (reproduces exp3).
"""
import numpy as np

alpha = 0.1
E = 1000                       # epoch length (steps)
alpha_v = 1.0 / (2 * E)        # v_slow window ~2 epochs (>= 1 epoch!)
lam = 1.0 / (2 * E)            # persistence EMA window ~2 epochs
beta2 = 0.999
N_EPOCHS = 200
T = E * N_EPOCHS


def run(snr, noise_kind, seed=0):
    rng = np.random.default_rng(seed)
    mu, sigma = snr, 1.0
    s_f = s_s = v_s = vhat = 0.0
    m_dsv = v_dsv = 0.0
    cp = []
    t = 0
    for ep in range(N_EPOCHS):
        if noise_kind == "epoch":
            noise = rng.standard_normal(E) * sigma
            noise = noise - noise.mean()        # sums to 0 over the epoch
        else:                                   # white (free walk)
            noise = rng.standard_normal(E) * sigma
        for i in range(E):
            t += 1
            g = mu + noise[i]
            vhat = beta2 * vhat + (1 - beta2) * g * g
            s_f = s_f + g
            d_sv = s_s - v_s
            m_dsv = (1 - lam) * m_dsv + lam * d_sv
            v_dsv = (1 - lam) * v_dsv + lam * d_sv * d_sv
            coh_p = min(max(m_dsv * m_dsv / (v_dsv + 1e-12), 0.0), 1.0)
            tick = alpha * s_f
            s_s += tick; s_f -= tick
            v_s += alpha_v * (s_s - v_s)
            if t > T - 10 * E:
                cp.append(coh_p)
    return float(np.mean(cp))


print(f"alpha={alpha} alpha_v={alpha_v:.5f} (~2 epochs) E={E} "
      f"N_EPOCHS={N_EPOCHS}\n")
print("  SNR     coh_persist(epoch-noise)   coh_persist(white control)")
for snr in (1.0, 0.3, 0.1, 0.03, 0.0):
    ce = run(snr, "epoch"); cw = run(snr, "white")
    tag = "  <- PURE NOISE" if snr == 0.0 else ""
    print(f"  {snr:5.2f}      {ce:7.4f}                  {cw:7.4f}{tag}")
ce_s = run(0.1, "epoch"); ce_n = run(0.0, "epoch")
print(f"\n  epoch-noise separation signal@0.1 / noise = "
      f"{ce_s:.4f}/{ce_n:.4f} = {ce_s/max(ce_n,1e-9):.1f}x")
