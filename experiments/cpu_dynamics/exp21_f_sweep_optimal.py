"""Exp 21: the F sweep at the optimal configuration (exp-20 champion).

Exp 20 fixed the configuration — NS drive at its own lr (1e-2), crop-aug,
gate on, window = ONE EPOCH, min-leak floor — and tested F at {0, 1.5}:
clean 97.45 (F=0) > 96.93 (F=1.5); noisy 95.60 (F=1.5) > 94.42 (F=0). This
fills the F axis at that configuration, INCLUDING the classically-forbidden
F >= 2 zone, runnable for the first time because the floor clamps survival
at min_leak (no ringing, no divergence — F past ~1.5 saturates on
incoherent coordinates instead of exploding).

Questions: where is noisy F*? Interior (~1-1.5) or rising past 1.5 (the
SDXL monotone-lambda pattern)? And is clean monotone-down (F*=0) as exp
12/16/20 suggest?

Grid: F in {0.1, 0.25, 0.5, 1.0, 2.5, 4.0} x noise {0, 30%}, 3 seeds,
4k x 25ep; F = 0 and 1.5 anchors from exp20_results.json (same seeds,
same protocol).
"""
import json
import time

from exp3_mnist import load_mnist
from exp20_window_floor_synthesis import run

F_LEVELS = (0.1, 0.25, 0.5, 1.0, 2.5, 4.0)

if __name__ == "__main__":
    data = load_mnist()
    mean = lambda v: sum(v) / len(v)
    spread = lambda v: (max(v) - min(v)) / 2
    e20 = json.load(open("exp20_results.json"))
    print("anchors (exp 20, W=1ep):")
    for key, label in (("gated|1|0.0|0.0", "F=0.0 clean"),
                       ("gated|1|0.0|1.5", "F=1.5 clean"),
                       ("gated|1|0.3|0.0", "F=0.0 noisy"),
                       ("gated|1|0.3|1.5", "F=1.5 noisy")):
        v = e20.get(key)
        if v:
            print(f"   {label}: deploy={v[0]*100:.2f}±{v[1]*100:.2f}%  "
                  f"memorized={v[2]*100:.1f}%")
    results = {}
    t0 = time.time()
    for nf in (0.0, 0.30):
        for fF in F_LEVELS:
            rs = [run("gated", 1, nf, fF, s, data) for s in (0, 1, 2)]
            tag = f"F={fF:.2f} noise={nf:.0%}"
            if any(r is None for r in rs):
                results[(nf, fF)] = None
                print(f"{tag}: DIVERGED   [{time.time()-t0:.0f}s]", flush=True)
                continue
            m, sp = mean([r[0] for r in rs]), spread([r[0] for r in rs])
            mf, mc = mean([r[1] for r in rs]), mean([r[2] for r in rs])
            results[(nf, fF)] = (m, sp, mf, mc)
            print(f"{tag}: deploy={m*100:.2f}±{sp*100:.2f}%  "
                  f"memorized={mf*100:.1f}%  coh={mc:.3f}   "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            json.dump({f"{n}|{f_}": v for (n, f_), v in results.items()},
                      open("exp21_results.json", "w"), indent=1)
