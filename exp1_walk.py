"""Experiment 1: does coherence-blind ACCEPTANCE diffuse noise into s_slow,
and does gating acceptance with a bootstrap floor bound it?

Routing per step (toy, scale_fwd=1):
    s_f += g                                  # free injection
    coh  = clip((alpha_v*(s_s - v_s))^2 / vhat, 0, 1)
    s_f -= kappa*(1-coh)*s_f                   # evaporation (within s_f)
    Delta = alpha*(coh + eps*(1-coh)) * s_f    # GATED acceptance w/ floor eps
    s_s += Delta; s_f -= Delta                 # exit into kept position
    v_s += alpha_v*(s_s - v_s)

eps = 1.0  -> acceptance = alpha (coherence-blind = current implementation).
eps -> 0   -> noise acceptance = alpha*eps, bounded at the source.

Measures:
  - NOISE ensemble (g ~ N(0,1)): std(s_s) vs t.  Ungated -> grows ~sqrt(t)
    (Var linear in t); gated-with-floor -> plateaus.
  - SIGNAL ensemble (g = +1 const): mean(s_s) vs t.  Must still integrate
    (~linear) for small eps, confirming the floor preserves the bootstrap.
"""
import numpy as np

alpha, alpha_v, kappa, beta2 = 0.1, 0.001, 0.05, 0.999
T, N = 80000, 400
checkpoints = [2500, 5000, 10000, 20000, 40000, 80000]


def run(eps_hi, kind, seed, eps_lo=None, cohpre_lam=None):
    # eps_lo=None -> constant eps_hi. Else linear anneal eps_hi -> eps_lo.
    # cohpre_lam set -> per-coord floor = coh_pre (EMA of coh, init 1),
    #   overrides eps. Self-terminating per coordinate.
    rng = np.random.default_rng(seed)
    s_f = np.zeros(N); s_s = np.zeros(N); v_s = np.zeros(N); vhat = np.zeros(N)
    coh_pre = np.ones(N)
    rec = {}
    for t in range(1, T + 1):
        g = rng.standard_normal(N) if kind == "noise" else np.ones(N)
        vhat = beta2 * vhat + (1 - beta2) * g * g
        s_f = s_f + g
        d_sv = s_s - v_s
        coh = np.clip((alpha_v * d_sv) ** 2 / (vhat + 1e-12), 0.0, 1.0)
        s_f = s_f - kappa * (1.0 - coh) * s_f
        if cohpre_lam is not None:
            floor = coh_pre
            coh_pre = (1.0 - cohpre_lam) * coh_pre + cohpre_lam * coh
        else:
            floor = eps_hi if eps_lo is None else \
                eps_lo + (eps_hi - eps_lo) * (1.0 - t / T)
        accept = alpha * (coh + floor * (1.0 - coh))
        Delta = accept * s_f
        s_s = s_s + Delta; s_f = s_f - Delta
        v_s = v_s + alpha_v * (s_s - v_s)
        if t in checkpoints:
            rec[t] = (float(np.std(s_s)), float(np.mean(s_s)),
                      float(np.mean(coh)))
    return rec


print(f"alpha={alpha} alpha_v={alpha_v} kappa={kappa}  N={N} coords  T={T}\n")
print("NOISE ensemble  -- std(s_s) vs t  [ungated eps=1 should grow ~sqrt(t)]")
hdr = "  eps     " + "".join(f"{t:>9}" for t in checkpoints)
print(hdr)
for eps in (1.0, 0.3, 0.1, 0.03):
    rec = run(eps, "noise", seed=0)
    row = "".join(f"{rec[t][0]:9.2f}" for t in checkpoints)
    print(f"  {eps:<6.2f}{row}")

print("\n  ratio std(80k)/std(10k)  [~2.83 if Var linear in t; ~1 if bounded]")
for eps in (1.0, 0.3, 0.1, 0.03):
    rec = run(eps, "noise", seed=0)
    print(f"  eps={eps:<5.2f} {rec[80000][0]/max(rec[10000][0],1e-9):5.2f}")

print("\nSIGNAL ensemble -- mean(s_s) vs t  [must integrate ~linearly; "
      "ideal full-accept ~ t]")
print(hdr)
for eps in (1.0, 0.3, 0.1, 0.03):
    rec = run(eps, "signal", seed=1)
    row = "".join(f"{rec[t][1]:9.0f}" for t in checkpoints)
    print(f"  {eps:<6.2f}{row}")

print("\nSIGNAL coherence(last) per eps  [should ->1 as d_sv builds]")
for eps in (1.0, 0.3, 0.1, 0.03):
    rec = run(eps, "signal", seed=1)
    print(f"  eps={eps:<5.2f} coh(80k)={rec[80000][2]:.3f}")

print("\n=== ANNEALED eps: 1.0 -> 0.03 (bootstrap hot, reject-noise cold) ===")
rn = run(1.0, "noise", seed=0, eps_lo=0.03)
rs = run(1.0, "signal", seed=1, eps_lo=0.03)
print("  NOISE std(s_s):  " + "".join(f"{rn[t][0]:9.2f}" for t in checkpoints))
print("  SIG  mean(s_s):  " + "".join(f"{rs[t][1]:9.0f}" for t in checkpoints))
print(f"  SIG  coh(80k)={rs[80000][2]:.3f}   "
      f"NOISE walk-ratio 80k/10k={rn[80000][0]/max(rn[10000][0],1e-9):.2f}")
print("  vs fixed eps=0.03: signal bootstrapped? compare coh and mean above")

print("\n=== coh_pre floor (per-coord, EMA of coh init 1; self-terminating) ===")
for lam in (0.003, 0.001, 0.0003):
    rn = run(0, "noise", seed=0, cohpre_lam=lam)
    rs = run(0, "signal", seed=1, cohpre_lam=lam)
    walk_ratio = rn[80000][0] / max(rn[10000][0], 1e-9)
    print(f"  lam={lam:<6.4f} NOISE std: " +
          "".join(f"{rn[t][0]:8.2f}" for t in checkpoints) +
          f"  ratio={walk_ratio:.2f}")
    print(f"             SIG  coh(80k)={rs[80000][2]:.3f}  "
          f"mean(s_s)80k={rs[80000][1]:.0f}")
print("  [want: NOISE ratio -> ~1 (BOUNDED), SIG coh high + mean ~full]")
