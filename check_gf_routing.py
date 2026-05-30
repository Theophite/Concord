"""Trivial check of gf-gated consolidation routing (pure numpy, no GPU).

Two independent weight coords driven for T steps:
  - 'coherent': constant gradient g=+1  (signal)
  - 'noisy'   : g ~ N(0,1) zero-mean    (pure noise, same power)

We want the router to:
  - read gf ~ 0 for coherent  -> consolidate into s_s (kept weight grows)
  - read gf ~ 1 for noisy     -> evaporate from s_f  (kept weight stays ~0)

Reports steady-state gf, s_f, s_s for each, plus the analytic-limit checks.
"""
import numpy as np

try:
    from prototype_packed_b import compute_drift_cancel_C
    C = float(compute_drift_cancel_C(0.1, 0.001))
    Csrc = "compute_drift_cancel_C(0.1,0.001)"
except Exception:
    C = 0.00908
    Csrc = "hardcoded fallback"

alpha   = 0.1     # consolidation (chase) rate
alpha_v = 0.001   # v_slow leak rate
kappa   = 0.05    # evaporation rate; < alpha so chase outruns it (bootstrap)
v_scale = 1.0
beta2   = 0.999   # v_hat EMA
T       = 60000
rng     = np.random.default_rng(0)

beta1_m = 0.9   # gradient-coherence EMA rate (m = EMA of g, same scale as g)

def run(kind, mode):
    s_f = s_s = v_s = vhat = m = 0.0
    hist_coh, hist_ss, hist_sf = [], [], []
    consolidated_total = 0.0
    for t in range(T):
        g = 1.0 if kind == "coherent" else float(rng.standard_normal())
        vhat = beta2 * vhat + (1 - beta2) * g * g
        m = beta1_m * m + (1 - beta1_m) * g
        s_f = s_f + g                                   # free injection (p=0)

        if mode == "m_ema":          # dedicated grad EMA gate, gated chase
            coh = np.clip(m * m / (vhat + 1e-12), 0.0, 1.0)
            t_c = alpha * coh * s_f; s_s += t_c; s_f -= t_c
            consolidated_total += t_c
            s_f -= kappa * (1.0 - coh) * s_f             # evaporate
        elif mode == "dsv":          # BUFFER-FREE: momentum = s_s - v_s,
                                     # unconditional chase, gated evaporation
            d_sv = s_s - v_s
            coh = np.clip((alpha_v * d_sv) ** 2 / (vhat + 1e-12), 0.0, 1.0)
            s_f -= kappa * (1.0 - coh) * s_f             # skim noise BEFORE chase
            t_c = alpha * s_f; s_s += t_c; s_f -= t_c    # unconditional chase
            consolidated_total += t_c
        v_s += alpha_v * (s_s - v_s)
        if t > T - 2000:
            hist_coh.append(coh); hist_ss.append(s_s); hist_sf.append(s_f)
    return (np.mean(hist_coh), s_s, s_f, np.mean(np.abs(hist_sf)),
            consolidated_total)

print(f"C (drift_cancel) = {C:.5f}  [{Csrc}]")
print(f"rates: alpha={alpha} alpha_v={alpha_v} kappa={kappa}  T={T}\n")
for mode in ("m_ema", "dsv"):
    print(f"--- gate mode: {mode} "
          + ("(dedicated grad EMA, +1 buffer)" if mode == "m_ema"
             else "(BUFFER-FREE: momentum = s_slow - v_slow)") + " ---")
    for kind in ("coherent", "noisy"):
        coh, s_s, s_f, sf_rms, cons = run(kind, mode)
        print(f"  [{kind:>8}] mean_coh(last2k)={coh:6.3f}  "
              f"s_slow={s_s:10.2f}  s_fast={s_f:8.3f}  "
              f"|s_fast|rms={sf_rms:7.3f}  consolidated={cons:10.2f}")

# Analytic-limit checks (forced gf), one step from a known state.
print("\nforced-limit checks (one routing step from s_f=10, s_s=v_s=0):")
for gf in (0.0, 1.0):
    s_f, s_s = 10.0, 0.0
    t_c = alpha * (1 - gf) * s_f; s_s2 = s_s + t_c; s_f2 = s_f - t_c
    t_e = kappa * gf * s_f2;       s_f2 = s_f2 - t_e
    print(f"  gf={gf}:  consolidate t_c={t_c:.3f} -> s_slow={s_s2:.3f},  "
          f"evaporate t_e={t_e:.3f} -> s_fast={s_f2:.3f}")
