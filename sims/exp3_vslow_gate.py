"""Does a LONG-timescale coherence separate signal from noise where the
short (chase-timescale) one couldn't?  Tests gating the v_slow leak.

coh_short = clip((alpha_v*d_sv)^2 / vhat, 0, 1)          [per-step-SNR limited]
coh_long  = clip((alpha_v*d_sv)^2 / (vhat*alpha_v), 0,1) [= coh_short/alpha_v]

We want, at realistic per-step SNR (<<1):
  - signal coords (drift mu, noise sigma): coh_long HIGH (gate open)
  - pure-noise coords (mu=0): coh_long LOW (no false positive from the walk)
If pure-noise coh_long is also high, the /alpha_v amplification let the
random-walk floor through -> the v_slow gate would admit noise.
"""
import numpy as np

alpha, alpha_v, beta2 = 0.1, 0.001, 0.999
T = 120000          # >> 1/alpha_v so v_slow/d_sv reach steady state


def run(snr, seed=0):
    rng = np.random.default_rng(seed)
    mu, sigma = snr, 1.0
    s_f = s_s = v_s = vhat = 0.0
    m_dsv = v_dsv = 0.0       # EMAs of d_sv and d_sv^2 (sign-persistence)
    lam = alpha_v             # persistence EMA rate (long timescale)
    cs, cl, cp = [], [], []
    for t in range(1, T + 1):
        g = mu + sigma * rng.standard_normal()
        vhat = beta2 * vhat + (1 - beta2) * g * g
        s_f = s_f + g
        d_sv = s_s - v_s
        m_dsv = (1 - lam) * m_dsv + lam * d_sv
        v_dsv = (1 - lam) * v_dsv + lam * d_sv * d_sv
        coh_s = min(max((alpha_v * d_sv) ** 2 / (vhat + 1e-12), 0.0), 1.0)
        coh_l = min(max((alpha_v * d_sv) ** 2 / (vhat * alpha_v + 1e-12),
                        0.0), 1.0)
        # sign-persistence: |E[d_sv]|^2 / E[d_sv^2] in [0,1] (Cauchy-Schwarz)
        coh_p = min(max(m_dsv * m_dsv / (v_dsv + 1e-12), 0.0), 1.0)
        tick = alpha * s_f                # unconditional chase (fit untouched)
        s_s += tick; s_f -= tick
        v_s += alpha_v * (s_s - v_s)
        if t > T - 5000:
            cs.append(coh_s); cl.append(coh_l); cp.append(coh_p)
    return float(np.mean(cs)), float(np.mean(cl)), float(np.mean(cp))


print(f"alpha={alpha} alpha_v={alpha_v}  T={T}\n")
print("  SNR     coh_short   coh_long   coh_persist  (want persist: sig->1 noise->0)")
for snr in (1.0, 0.3, 0.1, 0.03, 0.01, 0.0):
    cs, cl, cp = run(snr)
    tag = "  <- PURE NOISE" if snr == 0.0 else ""
    print(f"  {snr:5.2f}    {cs:7.4f}    {cl:7.4f}    {cp:7.4f}{tag}")
_, _, cp_sig = run(0.1); _, _, cp_noise = run(0.0)
print(f"\n  persist separation signal@0.1 / noise = "
      f"{cp_sig:.4f}/{cp_noise:.4f} = {cp_sig/max(cp_noise,1e-9):.1f}x")
