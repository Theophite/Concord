"""CPU test for NoiseScaleSeeder (concord_ot.py) -- the set-don't-hunt dissipation seeder.

Exec-extracts the class from source (the repo's pattern; concord_ot's module imports are
heavy) and runs it against fake layers whose gradients have KNOWN per-layer noise-to-signal
ratios. The accumulation writes replicate the apply-wrapper's contract exactly
(buf[:k] += g.flat[idx]; buf[k] += 1; buf[k+1] += ||g||^2 -- prototype_packed_2fast's
_lookup_nsr block); the wrapper itself is GPU-side and byte-trivial.

Pins: (1) construction registers a (buf, idx) pair per layer in the data_ptr-keyed registry;
(2) the NSR estimate tracks the known truth within sketch noise; (3) the MEDIAN seeded lam
equals lam0, and under the DEFAULT WHITENED law (lam = c/NSR, exp53) per-layer lams order
INVERSELY to true noise; (4) commits clamp to [lam_lo, 1.5]: the cleanest layer rails at
1.5, the noisiest floors at lam_lo, and land in gf_consol as lam/lr; (5) an under-filled
window is skipped and drained; (6) the once-per-t guard holds; (7) buffers are zeroed after
a commit window; (8) C RE-ANCHORS at every seeding (median lam returns to lam0 after a
global NSR shift), window readings blend into the running estimate at EW_BETA, and the
sketch coords redraw IN PLACE (same tensor object -- the graph-captured gather reads it by
pointer); (9) the smoothed NSRs round-trip through the sidecar and a fresh construction
seeds lams immediately (no flat interlude), and a sidecar whose saved clock is AHEAD
of the current run's is a foreign lineage and is REJECTED; (11) epoch_updates sets the
epoch-revisit floor (lam_lo = 4/updates-per-epoch) and commits respect it.

Run: python -m modules.util.optimizer.concord.tests.test_nsr_seeder_cpu
"""
import os
import sys
from types import SimpleNamespace as NS

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONCORD = os.path.normpath(os.path.join(_HERE, ".."))
if _CONCORD not in sys.path:
    sys.path.insert(0, _CONCORD)
_CONCORD_OT = os.path.normpath(os.path.join(_HERE, "..", "..", "concord_ot.py"))
SRC = open(_CONCORD_OT, encoding="utf-8").read()

from prototype_packed_2fast import _lookup_nsr  # noqa: E402


def _extract_class(name):
    lines = SRC.split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith(f"class {name}"))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i] and lines[i][0] not in " \t" and lines[i].startswith("class "):
            end = i
            break
    return "\n".join(lines[start:end])


_ns = {"torch": torch}
exec(_extract_class("NoiseScaleSeeder"), _ns)
NoiseScaleSeeder = _ns["NoiseScaleSeeder"]

_fails = []
def check(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
    if not cond:
        _fails.append(name)


def mk_layer(n, k):
    return NS(packed_w=torch.zeros(n, k, dtype=torch.int32), gf_consol=0.0)


LR, LAM0, EVERY = 1e-4, 0.4, 16
D_N, D_K = 64, 32
torch.manual_seed(7)

# three layers, known noise: sigma graded so NSR_true = D*sigma^2/||mu||^2 is well separated
layers = [mk_layer(D_N, D_K) for _ in range(3)]
mus = [torch.randn(D_N * D_K) * 0.5 for _ in range(3)]
sigmas = [0.2, 0.6, 1.8]
nsr_true = [float(D_N * D_K * s * s / mu.pow(2).sum()) for mu, s in zip(mus, sigmas)]

srv = NoiseScaleSeeder(layers, lr=LR, lam0=LAM0, every=EVERY, verbose=False)

# ── 1. construction registers a sketch per layer ──
check("1a registry holds every layer", all(_lookup_nsr(m.packed_w) is not None for m in layers))
k0 = layers[0]._nsr_idx.numel()
check("1b buffer layout k+2", all(m._nsr_buf.numel() == m._nsr_idx.numel() + 2 for m in layers),
      f"k={k0}")

def feed_into(ls, ms, ss, T, gen):
    """Replicate the apply-wrapper's accumulation contract for T micros."""
    for m, mu, s in zip(ls, ms, ss):
        buf, idx = m._nsr_buf, m._nsr_idx
        kk = idx.numel()
        for _ in range(T):
            g = mu + s * torch.randn(mu.numel(), generator=gen)
            buf[:kk] += g[idx]
            buf[kk] += 1.0
            buf[kk + 1] += float(g.pow(2).sum())


def feed(T, gen):
    feed_into(layers, mus, sigmas, T, gen)

gen = torch.Generator().manual_seed(0)
feed(EVERY * 2, gen)   # a full window of micros (2 micros/update here; > every*0.5 threshold)

# ── 2/3. commit: NSR accuracy, median anchoring, ordering, actuation ──
srv.step(EVERY)
check("2a C calibrated", srv.C is not None)
lams = [m.gf_consol * LR for m in layers]
med_lam = sorted(lams)[1]
check("3a median lam anchored at lam0", abs(med_lam - LAM0) < 1e-9, f"{med_lam:.4f} vs {LAM0}")
check("3b whitened law: lam orders INVERSELY to true noise", lams[0] > lams[1] > lams[2],
      "lams=" + ",".join(f"{x:.3f}" for x in lams))
# NSR estimate vs truth: read the estimator directly (backing it out of committed lams
# breaks at the clamp rails, which the whitened law engages on the cleanest layer)
nsr_hat = list(srv.nsr_ew)
ok_acc = all(0.4 < h / t < 2.5 for h, t in zip(nsr_hat, nsr_true))
check("2b NSR estimates within sketch noise of truth", ok_acc,
      " ".join(f"{h:.1f}/{t:.1f}" for h, t in zip(nsr_hat, nsr_true)))
check("7a buffers zeroed after the window", all(float(m._nsr_buf.abs().sum()) == 0.0 for m in layers))

# ── 6. once-per-t guard ──
g0 = layers[0].gf_consol
feed(EVERY * 2, gen)
srv.step(EVERY)        # same t as before -> must be a no-op
check("6a same-t re-step is a no-op", layers[0].gf_consol == g0)

# ── 5. under-filled window skipped and drained ──
for m in layers:
    m._nsr_buf.zero_()
feed(3, gen)           # only 3 micros < every*0.5
g_before = [m.gf_consol for m in layers]
srv.step(EVERY * 2)
check("5a under-filled window: no commit", [m.gf_consol for m in layers] == g_before)
check("5b under-filled window: drained", all(float(m._nsr_buf.abs().sum()) == 0.0 for m in layers))

# ── 4. clamp rails ──
# NOTE the estimator ceiling: the sigma^2/T bias saturates NSR readings at ~T-1 micros,
# so the extreme layer only out-ratios the median (and hits the 1.5 rail) with a window
# long enough for its true NSR to register -- feed 128 micros here (production windows are
# thousands of micros; the seeder docstring documents the constraint).
mus.append(torch.randn(D_N * D_K) * 0.5)
sigmas.append(50.0)    # absurd noise -> lam must rail at 1.5 (exp52: 1.7+ diverges under the bracket arm)
layers.append(mk_layer(D_N, D_K))
srv2 = NoiseScaleSeeder(layers, lr=LR, lam0=LAM0, every=EVERY, verbose=False)
gen2 = torch.Generator().manual_seed(1)
feed(128, gen2)
srv2.step(EVERY)
check("4a whitened: extreme-noise layer gets the coldest lam",
      layers[3].gf_consol * LR == min(m.gf_consol * LR for m in layers)
      and layers[3].gf_consol * LR < 0.1, f"lam={layers[3].gf_consol * LR:.3f}")
check("4b whitened: cleanest layer rails at lam=1.5",
      abs(layers[0].gf_consol * LR - 1.5) < 1e-9, f"lam={layers[0].gf_consol * LR:.3f}")
check("4c no layer below the floor", all(m.gf_consol * LR >= 0.02 - 1e-12 for m in layers))

# ── 8. re-anchor + EW blending + in-place sketch redraw ──
# Double every sigma (true NSRs ~4x, a global shift). Re-anchor: the median lam must RETURN
# to lam0 (pure-shape reallocation, no drift with global NSR). EW: the smoothed estimate
# moves only partway toward the new window (BETA=0.4 -> ~2.2x on a 4x shift). Redraw: the
# sketch coords change under the SAME tensor object (graph-captured gather reads by pointer).
ew_before = list(srv2.nsr_ew)
idx_objs = [m._nsr_idx for m in layers]
idx_vals = [m._nsr_idx.clone() for m in layers]
feed_into(layers, mus, [s * 2.0 for s in sigmas], 128, gen2)
srv2.step(EVERY * 2)
# the anchor pins the MEDIAN-NSR layer's lam at lam0 (with rails engaged, the sorted-lams
# median is not the anchored quantity)
med_i = sorted(range(len(layers)), key=lambda i: srv2.nsr_ew[i])[len(layers) // 2]
check("8a median-NSR layer re-anchors at lam0 after a global NSR shift",
      abs(layers[med_i].gf_consol * LR - LAM0) < 1e-9,
      f"{layers[med_i].gf_consol * LR:.4f} vs {LAM0}")
r = srv2.nsr_ew[1] / ew_before[1]
check("8b window blends at EW_BETA (partial move on a 4x shift)", 1.2 < r < 3.9,
      f"ratio={r:.2f} (expect ~2.2)")
check("8c sketch redraw is in place: same tensor, new coords",
      all(m._nsr_idx is o for m, o in zip(layers, idx_objs))
      and all(not torch.equal(m._nsr_idx, v) for m, v in zip(layers, idx_vals)))

# ── 9. sidecar round-trip: restore -> immediate seed (no flat interlude) ──
import tempfile
sc = os.path.join(tempfile.gettempdir(), "test_concord_nsr_sidecar.json")
if os.path.exists(sc):
    os.remove(sc)
layers3 = [mk_layer(D_N, D_K) for _ in range(3)]
srv3 = NoiseScaleSeeder(layers3, lr=LR, lam0=LAM0, every=EVERY, sidecar=sc, verbose=False)
gen3 = torch.Generator().manual_seed(2)
feed_into(layers3, mus[:3], sigmas[:3], EVERY * 2, gen3)
srv3.step(EVERY)
check("9a sidecar written after a commit window", os.path.exists(sc))
layers4 = [mk_layer(D_N, D_K) for _ in range(3)]
srv4 = NoiseScaleSeeder(layers4, lr=LR, lam0=LAM0, every=EVERY, sidecar=sc, verbose=False)
check("9b smoothed NSRs survive the round-trip",
      all(a is not None and b is not None and abs(a - b) < 1e-12
          for a, b in zip(srv3.nsr_ew, srv4.nsr_ew)))
check("9c fresh construction seeds lams immediately from the sidecar",
      all(abs(m4.gf_consol - m3.gf_consol) < 1e-6 for m3, m4 in zip(layers3, layers4)))
# 9d: a sidecar from a NEWER clock lineage (fresh run in an old workspace) is rejected
import json as _json
d9 = _json.load(open(sc))
d9["step"] = 100000
_json.dump(d9, open(sc, "w"))
layers4b = [mk_layer(D_N, D_K) for _ in range(3)]
srv4b = NoiseScaleSeeder(layers4b, lr=LR, lam0=LAM0, every=EVERY, sidecar=sc,
                         now_t=0, verbose=False)
check("9d foreign-lineage sidecar rejected (saved t >> current t)",
      all(v is None for v in srv4b.nsr_ew))
os.remove(sc)

# ── 11. epoch-revisit floor ──
layers6 = [mk_layer(D_N, D_K) for _ in range(3)]
srv6 = NoiseScaleSeeder(layers6, lr=LR, lam0=LAM0, every=EVERY, epoch_updates=100,
                        verbose=False)
check("11a lam_lo = 4/updates-per-epoch", abs(srv6.lam_lo - 0.04) < 1e-12,
      f"lam_lo={srv6.lam_lo}")
gen6 = torch.Generator().manual_seed(4)
feed_into(layers6, mus[:3], sigmas[:3], EVERY * 2, gen6)
srv6.step(EVERY)
check("11b commits respect the epoch floor",
      all(m.gf_consol * LR >= 0.04 - 1e-12 for m in layers6))

# ── 12. per-row arm meter (nsr_per_row) ──
def pack(eL, eH):
    """Pack known int8 arms into the 2-fast word layout (s_slow = v_slow = 0)."""
    return ((eL.to(torch.int32) & 0xFF) << 24) | ((eH.to(torch.int32) & 0xFF) << 16)


def mk_row_layer(n, k, eL, eH):
    m = mk_layer(n, k)
    m.packed_w = pack(eL, eH)
    m.arm_col_exp = torch.zeros(k, dtype=torch.int32)
    m.col_exp = torch.zeros(k, dtype=torch.int32)
    return m


# rows: (a) corroborated eL=eH=6 -> R~0, occ 6; (b) contested eL=6,eH=-6 -> R->clamp
# 0.999, occ 6; (c) residue eL=eH=0 except one +-1 -> occ ~0 (masked)
eLr = torch.zeros(3, D_K, dtype=torch.int32)
eHr = torch.zeros(3, D_K, dtype=torch.int32)
eLr[0], eHr[0] = 6, 6
eLr[1], eHr[1] = 6, -6
eLr[2, 0], eHr[2, 0] = 1, 1
mrow = mk_row_layer(3, D_K, eLr, eHr)
R12, occ12 = NoiseScaleSeeder._row_arm_stats(mrow)
check("12a arm stats: corroborated row reads R~0, contested reads R~1, occ correct",
      R12[0] < 0.01 and R12[1] > 0.99 and abs(float(occ12[0]) - 6.0) < 1e-6
      and float(occ12[2]) < 0.1,
      f"R={[round(float(x), 4) for x in R12]} occ={[round(float(x), 2) for x in occ12]}")

# per-row commit end-to-end: many readable rows (the >=100 gate), two layers with
# opposite R so the row lams differentiate; the residue layer keeps the layer lam
NROW = 64
eL_a = torch.full((NROW, D_K), 6, dtype=torch.int32)   # corroborated -> low NSR -> HOT lam
mA = mk_row_layer(NROW, D_K, eL_a, eL_a.clone())
eL_b = torch.full((NROW, D_K), 6, dtype=torch.int32)   # contested -> high NSR -> COLD lam
mB = mk_row_layer(NROW, D_K, eL_b, -eL_b)
mC = mk_row_layer(NROW, D_K, torch.zeros(NROW, D_K, dtype=torch.int32),
                  torch.zeros(NROW, D_K, dtype=torch.int32))   # residue -> masked
rows_layers = [mA, mB, mC]
srv7 = NoiseScaleSeeder(rows_layers, lr=LR, lam0=LAM0, every=EVERY, per_row=True,
                        verbose=False)
check("12b per-row arming widens gf bufs to [N] pre-capture",
      all(m._gf_consol_buf.numel() == NROW for m in rows_layers))
gen7 = torch.Generator().manual_seed(5)
# PHASE GATE: rows commit only after ROW_PERSIST consecutive resolvable windows --
# run enough windows for the corroborated rows to earn eligibility
for _w in range(1, NoiseScaleSeeder.ROW_PERSIST + 1):
    feed_into(rows_layers, mus[:3], sigmas[:3], EVERY * 2, gen7)
    srv7.step(EVERY * _w)
lamA = mA._gf_consol_buf * LR
lamB = mB._gf_consol_buf * LR
lamC = mC._gf_consol_buf * LR
# with half the readable rows resolvable, the global row median lands INSIDE the
# resolvable population -> eligible rows commit ~ the anchor lam0 (the anchor property),
# DIFFERENTIATED from their own layer's (railed) lam -- and never above the cap
capA = min(1.5, NoiseScaleSeeder.ROW_CAP_MULT * srv7._layer_lam[0])
check("12c persistent rows commit differentiated (anchor-ish, != layer lam, <= cap)",
      abs(float(lamA.median()) - LAM0) < 0.05
      and abs(float(lamA.median()) - srv7._layer_lam[0]) > 0.5
      and float(lamA.max()) <= capA + 1e-6,
      f"med={float(lamA.median()):.4f} layer={srv7._layer_lam[0]:.3f} cap={capA:.3f}")
check("12c2 saturated (contested) rows hold the LAYER lam, never a hot fluke",
      abs(float(lamB.median()) - srv7._layer_lam[1]) < 1e-6,
      f"{float(lamB.median()):.4f} vs layer {srv7._layer_lam[1]:.4f}")
check("12d residue rows keep the layer lam",
      abs(float(lamC.median()) - srv7._layer_lam[2]) < 1e-6,   # fp32 buf round-trip
      f"{float(lamC.median()):.6f} vs layer {srv7._layer_lam[2]:.6f}")
# one-window flukes must NOT commit: fresh seeder, single window -> row buf untouched
# (NS fakes have no gf_consol property, so 'untouched' = the arming value, byte-for-byte)
srv8 = NoiseScaleSeeder([mk_row_layer(NROW, D_K, eL_a, eL_a.clone())], lr=LR, lam0=LAM0,
                        every=EVERY, per_row=True, verbose=False)
buf8_before = srv8.layers[0]._gf_consol_buf.clone()
gen8 = torch.Generator().manual_seed(6)
feed_into(srv8.layers, mus[:1], sigmas[:1], EVERY * 2, gen8)
srv8.step(EVERY)
check("12e persistence: a single resolvable window commits no row values",
      torch.equal(srv8.layers[0]._gf_consol_buf, buf8_before))

n_checks = 27
print(f"\n{'ALL PASS' if not _fails else 'FAILED: ' + ', '.join(_fails)} "
      f"({n_checks - len(_fails)}/{n_checks} checks)")
sys.exit(1 if _fails else 0)
