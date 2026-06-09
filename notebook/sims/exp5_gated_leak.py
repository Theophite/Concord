"""(a) Gate the v_slow leak by coh_persist and check the kept anchor is
DENOISED: signal coords -> v_slow tracks the drift; pure-noise coords ->
v_slow stays clean (rejects the within-epoch bridge), much smaller than the
ungated leak.
(b) Sweep the leak/persistence window across the epoch boundary -> the
denoising should only work when window >= 1 epoch (epoch-noise has cancelled).

Gated leak:   v_s += alpha_v * coh_persist * (s_s - v_s)
coh_persist = |EMA(d_sv)|^2 / EMA(d_sv^2),  EMAs at rate lam = alpha_v.
"""
import numpy as np

alpha = 0.1
E = 1000                 # epoch length
beta2 = 0.999
N_EPOCHS = 240
N = 200                  # ensemble size


def run(window_epochs, kind, gated, seed=0):
    rng = np.random.default_rng(seed)
    av = 1.0 / (window_epochs * E)            # v_slow + persistence rate
    mu = 0.1 if kind == "signal" else 0.0     # SNR=0.1 signal vs pure noise
    s_f = np.zeros(N); s_s = np.zeros(N); v_s = np.zeros(N)
    vhat = np.zeros(N); m_d = np.zeros(N); v_d = np.zeros(N)
    for ep in range(N_EPOCHS):
        noise = rng.standard_normal((E, N))
        noise -= noise.mean(axis=0, keepdims=True)   # zero-sum per epoch
        for i in range(E):
            g = mu + noise[i]
            vhat = beta2 * vhat + (1 - beta2) * g * g
            s_f = s_f + g
            d = s_s - v_s
            m_d = (1 - av) * m_d + av * d
            v_d = (1 - av) * v_d + av * d * d
            coh = np.clip(m_d * m_d / (v_d + 1e-12), 0.0, 1.0)
            s_s = s_s + alpha * s_f; s_f = s_f - alpha * s_f   # chase (fit)
            gate = coh if gated else 1.0
            v_s = v_s + av * gate * (s_s - v_s)
    return float(np.mean(np.abs(v_s))), float(np.mean(np.abs(s_s)))


print(f"alpha={alpha} E={E} (epoch)  N_EPOCHS={N_EPOCHS}  ens={N}\n")
print("  window   sig|v_s|/|s_s|   noise|v_s|(gated)  noise|v_s|(ungated)  "
      "denoise x")
for w in (0.5, 1.0, 2.0, 4.0):
    sg_v, sg_s = run(w, "signal", gated=True)
    nz_g, _ = run(w, "noise", gated=True)
    nz_u, _ = run(w, "noise", gated=False)
    track = sg_v / max(sg_s, 1e-9)
    denoise = nz_u / max(nz_g, 1e-9)
    print(f"  {w:4.1f}ep   {track:8.3f}        {nz_g:9.3f}         "
          f"{nz_u:9.3f}        {denoise:5.1f}x")
print("\n  want: track ~1 (signal kept), denoise >> 1 (noise rejected),")
print("        and denoise should jump once window >= 1 epoch.")
