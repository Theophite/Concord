"""Experiment 2: what coherence does the (alpha_v*d_sv)^2/vhat gate produce
as a function of gradient SNR?  If realistic (low) SNR -> low coh, the
coh_pre acceptance gate over-throttles (explains the CIFAR underfit).

One coord, g_t = mu + sigma*xi  (xi~N(0,1)); SNR = mu/sigma. Run the
unconditional-chase dynamics to steady state, measure coh.
"""
import numpy as np

alpha, alpha_v, beta2 = 0.1, 0.001, 0.999
T = 60000


def steady_coh(snr, seed=0):
    rng = np.random.default_rng(seed)
    mu, sigma = snr, 1.0          # fix noise=1, vary drift=SNR
    s_f = s_s = v_s = vhat = 0.0
    cohs = []
    for t in range(1, T + 1):
        g = mu + sigma * rng.standard_normal()
        vhat = beta2 * vhat + (1 - beta2) * g * g
        s_f = s_f + g
        d_sv = s_s - v_s
        coh = min(max((alpha_v * d_sv) ** 2 / (vhat + 1e-12), 0.0), 1.0)
        tick = alpha * s_f                # unconditional chase (build d_sv)
        s_s += tick; s_f -= tick
        v_s += alpha_v * (s_s - v_s)
        if t > T - 3000:
            cohs.append(coh)
    return float(np.mean(cohs))


print(f"alpha={alpha} alpha_v={alpha_v}  T={T}")
print("\n  SNR=mu/sigma   steady coh   acceptance floor it implies")
print("  (CIFAR per-coord SNR is typically << 1)")
for snr in (10.0, 3.0, 1.0, 0.3, 0.1, 0.03, 0.01):
    c = steady_coh(snr)
    # acceptance for an incoherent-looking coord once coh_pre decays to ~coh:
    print(f"   {snr:6.2f}       {c:7.4f}      ~alpha*coh = {alpha*c:.4f} "
          f"(vs alpha={alpha} ungated)")
