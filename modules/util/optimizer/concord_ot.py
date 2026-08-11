"""OneTrainer <-> Concord glue.

Concord is NOT a torch.optim.Optimizer. It swaps the UNet's nn.Linear/nn.Conv2d for
packed self-stepping layers whose optimizer update is fused INTO the autograd backward
(there is no optimizer.step() for them). Around that it needs two per-step callbacks:
  - BEFORE the forward/backward: winner_step() advances the lr / noise-sigma / coherence
    -floor schedule (these live in device tensors the fused backward reads);
  - AFTER the update: a gated rebalance() (fires only when a packed mantissa actually
    overflows -- ~0% of steps at finetune lr, so nearly free).

So the OneTrainer-visible "optimizer" for the CONCORD choice is just a plain SGD over
the NON-swapped (aux) params -- norms, biases, embeddings -- and this controller carries
the Concord half. One controller per run, stored on the model; the trainer calls
before_step()/after_step() around its existing loop.
"""
import sys
from pathlib import Path

import torch

# the vendored Concord core lives next to this file
_CONCORD_DIR = str((Path(__file__).parent / "concord").resolve())
if _CONCORD_DIR not in sys.path:
    sys.path.insert(0, _CONCORD_DIR)


def _resolve_single_token_ids(tokenizer, words):
    """Resolve words to vocab ids, keeping only those that are a SINGLE token. A
    multi-token word (e.g. 'tok' -> 'pen','is') can't be zeroed without breaking the
    shared subwords, so it's skipped. Returns (ids, skipped_words)."""
    ids, skipped = [], []
    for w in words:
        toks = tokenizer(w, add_special_tokens=False).input_ids
        if len(toks) == 1:
            ids.append(int(toks[0]))
        else:
            skipped.append(w)
    return ids, skipped


class SanitizePlane:
    """Independent control plane: zero the embedding rows of given single-token vocab
    words in BOTH SDXL text encoders, and keep them zeroed across steps. The saved model
    then embeds those words to ~nothing at inference (standard CLIP tokenizer + modified
    weights -> works in any SDXL tool). Works with any optimizer (not tied to Concord)."""

    def __init__(self, model, tokens_csv: str):
        words = [t.strip() for t in tokens_csv.split(",") if t.strip()]
        self.ids1, sk1 = _resolve_single_token_ids(model.tokenizer_1, words)
        self.ids2, sk2 = _resolve_single_token_ids(model.tokenizer_2, words)
        self.skipped = sorted(set(sk1) & set(sk2))   # skipped in BOTH (truly multi-token)
        self.reapply(model)
        msg = f"[concord] sanitize: zeroed {len(self.ids1)} (CLIP-L) / {len(self.ids2)} (CLIP-G) token rows"
        if self.skipped:
            msg += f" | skipped multi-token (can't zero a subword): {self.skipped}"
        print(msg)

    @torch.no_grad()
    def reapply(self, model):
        for te, ids in ((model.text_encoder_1, self.ids1), (model.text_encoder_2, self.ids2)):
            if ids:
                w = te.get_input_embeddings().weight
                w[ids] = 0.0


def make_concord_config(learning_rate: float, optimizer_config=None):
    """Map OneTrainer settings onto the validated winner config: lr comes from the main
    learning_rate field; the winner knobs (gf_consol/noise/sigmag_peak/ratio_coh/warmup/
    lr_min_frac) come from the GUI optimizer-params panel (None -> validated winner default)."""
    from concord_winner import ConcordConfig
    d = ConcordConfig()

    def pick(name, default):
        v = getattr(optimizer_config, name, None) if optimizer_config is not None else None
        return default if v is None else v

    # Leak floor: DERIVED from the chase floor -- deliberately NOT an independent option.
    # Exp 73 (experiments/cpu_dynamics, 2026-07-18): the two gates are CALIBRATION
    # PARTNERS -- under drift mu/u = L*alpha*gc/gl, so C* fixes the chase:leak rate
    # RATIO, not either floor alone. A leak floor far off the chase floor decalibrates
    # the drift meter (10x off = certification deadlock at every drift rate, the
    # ungated-arm result; the pre-derivation production state, chase_min 0.01 vs pinned
    # leak_min 0.1, was exactly that mis-ratio). The user sets the CHASE floors (the
    # real dial: admission selectivity) and the leak FOLLOWS:
    #     leak_min   = chase_min      (the winner's own pairing, ratio 0.9 -- dead
    #                                  center of exp73's wide ignition tolerance)
    #     leak_start = 0.999          (validated withhold-judgment-early ignition
    #                                  design, independent of the chase start)
    # Winner-default chase (0.9, 0.1) therefore yields the validated winner leak
    # (0.999, 0.1) BYTE-IDENTICALLY. Chase endpoints are clamped to [0,1] (a floor > 1
    # flips the affine gate into coherence-INVERTING). A leak value still carried by an
    # old config JSON is IGNORED (loud note below). There is NO override: the
    # CONCORD_LEAK_FLOOR env hatch was removed at the user's request (2026-07-18,
    # "remove the option for it to be any different") -- a mis-ratioed leak floor has
    # exactly one observed behavior class (exp73 deadlock) and zero legitimate uses.
    # A set-but-ignored env var warns loudly rather than silently doing nothing.
    # Ablation note for future researchers: mis-ratio dynamics remain reproducible on
    # the CPU reference (exp73's arms); the production path deliberately cannot.
    import os as _os
    _gc_raw = (float(pick("ratio_chase_floor", d.ratio_chase_floor)),
               float(pick("ratio_chase_floor_min", d.ratio_chase_floor_min)))
    _gc = (min(1.0, max(0.0, _gc_raw[0])), min(1.0, max(0.0, _gc_raw[1])))
    if _gc != _gc_raw:
        print(f"[concord] chase floor {_gc_raw} CLAMPED to {_gc} (floors live in [0,1]; "
              f"a floor > 1 inverts the coherence gate)", flush=True)
    _lf = (d.ratio_leak_floor, _gc[1])                  # derived: (0.999, chase_min)
    if _os.environ.get("CONCORD_LEAK_FLOOR", ""):
        print(f"[concord] WARNING: CONCORD_LEAK_FLOOR is set but NO LONGER SUPPORTED -> "
              f"ignored; the leak floor is always derived from the chase floor ({_lf}). "
              f"Remove the env var from your launcher.", flush=True)
    _lf_cfg = (pick("ratio_leak_floor", None), pick("ratio_leak_floor_min", None))
    if any(v is not None and float(v) != w for v, w in zip(_lf_cfg, _lf)):
        print(f"[concord] NOTE: config carries ratio_leak_floor={_lf_cfg} -> IGNORED; "
              f"the leak floor is DERIVED from the chase floor: {_lf} (exp73 ratio law, "
              f"leak_min = chase_min)", flush=True)

    return ConcordConfig(
        lr=float(learning_rate),
        gf_consol=float(pick("gf_consol", d.gf_consol)),
        noise=bool(pick("noise", d.noise)),
        sigmag_peak=float(pick("sigmag_peak", d.sigmag_peak)),
        ratio_coh=bool(pick("ratio_coh", d.ratio_coh)),
        lazy_gate=bool(pick("lazy_gate", d.lazy_gate)),
        lazy_active_thresh=float(pick("lazy_active_thresh", d.lazy_active_thresh)),
        warmup=int(pick("warmup", d.warmup)),
        lr_min_frac=float(pick("lr_min_frac", d.lr_min_frac)),
        step_cap=float(pick("step_cap", d.step_cap)),
        gf_trust_delta_sq=float(pick("gf_trust_delta_sq", d.gf_trust_delta_sq)),
        nsr_per_row=bool(pick("nsr_per_row", d.nsr_per_row)),
        min_leak=float(pick("min_leak", d.min_leak)),
        evap_build_min=float(pick("evap_build_min", d.evap_build_min)),
        lamb_trust=bool(pick("lamb_trust", d.lamb_trust)),
        lamb_cap=float(pick("lamb_cap", d.lamb_cap)),
        lamb_clip=float(pick("lamb_clip", d.lamb_clip)),
        beta2=float(pick("beta2", d.beta2) or d.beta2),
        beta2_epoch_window=bool(pick("beta2_epoch_window", d.beta2_epoch_window)),
        vhat_warmstart=bool(pick("vhat_warmstart", d.vhat_warmstart)),
        bias_correct_v=bool(pick("bias_correct_v", d.bias_correct_v)),
        coh_vhat=bool(pick("coh_vhat", d.coh_vhat)),
        coh_kappa=float(pick("coh_kappa", d.coh_kappa)),
        dissipation_fill_ramp=bool(pick("dissipation_fill_ramp", d.dissipation_fill_ramp)),
        telescope_epoch_window=bool(pick("telescope_epoch_window", d.telescope_epoch_window)),
        autotune_table=pick("autotune_table", d.autotune_table),
        autotune_beta1_on=float(pick("autotune_beta1_on", d.autotune_beta1_on)),
        autotune_beta1_coh=float(pick("autotune_beta1_coh", d.autotune_beta1_coh)),
        autotune_reprobe_band=pick("autotune_reprobe_band", d.autotune_reprobe_band),
        autotune_gamma_snr=pick("autotune_gamma_snr", d.autotune_gamma_snr),
        autotune_gamma_snr_on=bool(pick("autotune_gamma_snr_on", d.autotune_gamma_snr_on)),
        autotune_servo_per_epoch=int(pick("autotune_servo_per_epoch", d.autotune_servo_per_epoch)),
        concord_conv_full_vhat=bool(pick("concord_conv_full_vhat", d.concord_conv_full_vhat)),
        concord_evap_slack=float(pick("concord_evap_slack", d.concord_evap_slack)),
        concord_train_cond_embed=bool(pick("concord_train_cond_embed", d.concord_train_cond_embed)),
        dissipation=pick("dissipation", d.dissipation),
        two_fast=bool(pick("two_fast", d.two_fast)),
        bracket_d=float(pick("bracket_d", d.bracket_d)),
        evict_valve=bool(pick("evict_valve", d.evict_valve)),
        evict_gain=float(pick("evict_gain", d.evict_gain)),
        evict_cf_gate=bool(pick("evict_cf_gate", d.evict_cf_gate)),
        heldout_router=bool(pick("heldout_router", d.heldout_router)),
        router_coh_noise=bool(pick("router_coh_noise", d.router_coh_noise)),
        perfcoh_partition=bool(pick("perfcoh_partition", d.perfcoh_partition)),
        perfcoh_tau=float(pick("perfcoh_tau", d.perfcoh_tau)),
        noise_seed_servo=bool(pick("noise_seed_servo", d.noise_seed_servo)),
        kaiming_init=bool(pick("kaiming_init", d.kaiming_init)),
        kaiming_scale=float(pick("kaiming_scale", d.kaiming_scale)),
        chase_epoch_window=bool(pick("chase_epoch_window", d.chase_epoch_window)),
        chase_alpha=float(pick("chase_alpha", d.chase_alpha)),
        alpha_v_fast=float(pick("alpha_v_fast", d.alpha_v_fast)),
        ratio_chase_floor=_gc[0],                # clamped [0,1] (see the derivation block)
        ratio_chase_floor_min=_gc[1],
        ratio_leak_floor=float(_lf[0]),          # DERIVED from chase (see block above)
        ratio_leak_floor_min=float(_lf[1]),
    )


class NoiseScaleSeeder:
    """Set-don't-hunt dissipation (exp49/exp50 [epic-williamson lineage]): SEED each layer's
    kappa from its MEASURED gradient noise-to-signal ratio instead of hunting with a secant.
    LAW (exp53): WHITENED evidence rate, lam_layer = c / NSR_layer -- friction
    grants each layer the integration window its noise requires, so every layer's arms
    equilibrate at the same steady-state SNR^2 = 2/c and the commitment gate reads one
    evidence quality everywhere. exp53 (CPU, MNIST, 3 seeds): whitened passed all four
    criteria (late-extraction, memorization defense, epoch cliff, early no-regression);
    the REMOVED assigned law lam = C * NSR LEAKED
    memorization vs even flat (mem 15.3% vs 9.8% at 30% noise; 20.3% vs 12.6% at 10%) --
    temporally-coherent drift reads clean and the assigned law rewards it with the
    patient window. NSR = E||g||^2 / ||E g||^2 - 1, from the RAW gradient stream (exp50:
    calibrates better than the preconditioned stream). ||E g||^2 is estimated from a
    k-coordinate mean-gradient sketch scaled by D/k. SMOOTHING: single-window estimates are too noisy to commit raw -- adjacent
    production windows moved the layer-MEDIAN 2x and single windows produce sub-1 lows and
    sketch-luck highs beyond the T-1 ceiling -- so each window's reading blends into a
    per-layer running estimate (EW_BETA), the sketch coordinates are REDRAWN every window
    (a fixed sketch can sit on, or miss, the heavy coordinates of a structured mean
    indefinitely; that bias never averages out without redraw), and lams commit from the
    smoothed values. The measurement is a k-coordinate gather plus one norm per backward --
    cheap enough that cadence is not a cost consideration. ANCHOR: C re-derives at EVERY
    seeding from the smoothed median, so the median layer always runs exactly the
    configured lam and the law is PURE-SHAPE reallocation (calibrate-once let the median
    drift with global NSR inside a process and made C a per-process lottery of the
    calibration window). PERSISTENCE: the smoothed NSRs round-trip through a workspace
    sidecar keyed by layer name -- NSR is a property of data+arch shared across segments,
    and without the sidecar every relaunch ran flat until its first window filled.
    Commits clamp to lam in [lam_lo, 1.5]. CEILING 1.5: the divergence wall is lam*(1+d)
    ~ 2, and the rate bracket's punished arm runs lam*(1+bracket_d) -- at the default
    d=0.25 a committed 1.85 puts that arm at 2.31 and the weights overflow (exp52
    [epic-williamson] measured the boundary empirically: 1.5/1.6 bounded, 1.7/1.85
    exploded; worse, a runaway reads as a huge coherent mean so NSR collapses and the
    seed chases the explosion DOWN in lam -- self-reinforcing). 1.5 is safe for every d
    in [0, 0.33]. FLOOR lam_lo = 4/updates-per-epoch (window <= epoch/4), the
    router-defense horizon rather than taste: the corroboration window 1/lam must stay
    well inside the dataset revisit period, or a RECURRING example's pushes straddle both
    arms within one window and self-corroborate -- the held-out defense breaks
    structurally (exp53 cliff cell: window = 2 epochs took mem 5% -> 55% and gen -5.4pp).
    Falls back to the legacy 0.02 when the epoch length is unknown. CONVERGENCE CONFOUND (exp52):
    a CONVERGED layer's residual mean-gradient vanishes, so its NSR reads high and its lam
    drifts to the ceiling regardless of true noise -- harmless post-convergence (the answer
    is already consolidated; high lam just taxes new arm content) but the committed lam is
    NOT a noise readout for converged layers. ESTIMATOR CEILING: the sigma^2/T bias in the
    mean-gradient estimate saturates a single window's NSR reading at ~T-1 (T = micros in
    the window), so readings are trustworthy only while true NSR << window micros; do not
    shrink the window below ~32 updates. CPU receipts: the one-point-calibrated seed
    never underperformed an exhaustive lam grid across noise fractions 10-50% and accum 1-4,
    and out-resolved the grid where the ridge fell between grid points (exp50). Under the
    held-out router the arm gap is a cross-split data meter, not dW/dlam -- never wire a
    dissipation controller to it. This seeder is the ONLY dissipation controller (the
    hunting servos were removed); it owns per-layer kappa."""
    per_layer = True
    K = 128           # sketch coords per layer (D/k extrapolation; 128 was CPU-sufficient)
    EW_BETA = 0.4     # per-window blend: steady-state variance factor B/(2-B)=0.25 (sd
                      # halves vs raw windows) at ~2.5-window lag -- slow enough to absorb
                      # window-position swings, fast enough to track real drift
    SIDECAR_VER = 1
    # ── per-row law (nsr_per_row): the ARM meter ──
    # exp54 REFUTED the gradient column-sketch as a per-row NSR estimator (rank-corr vs
    # exact 0.07-0.28 at HALF the columns; the row law seeded from it was WORSE than the
    # layer law). The per-row meter here is the ROUTER ARMS instead: R_row =
    # sum(gap_w^2)/sum(sum_w^2) over the row (gap = e_L - e_H, cross-split disagreement)
    # -- full-row, integrated continuously by the kernel itself, no extra gradient
    # plumbing. Mapping: NSR_row ~ R/(1-R); law: lam_row = c/NSR_row (whitened),
    # anchored at the GLOBAL row median (per-layer anchoring would reintroduce
    # layer-blindness). KNOWN CLOSED LOOP: R reads state that depends on lam (hot lam
    # thins arms -> quantization residue -> R -> 1); mitigations: rows below ROW_OCC_MIN
    # mean arm occupancy are UNREADABLE (they keep the layer lam), EW smoothing, and the
    # per-seeding drift is PRINTED so divergence is visible. exp56 (CPU closed-loop
    # validation) is IN FLIGHT; this shipped ahead of its verdict by explicit user call
    # -- fix-forward on its results. Per-row state is NOT persisted (rebuilds in ~2-3
    # windows after a restart; masked/unseeded rows run the layer lam meanwhile).
    ROW_OCC_MIN = 0.25   # mean |arm| ints below this = residue; R is the ruler, not the row
    ROW_EW_BETA = 0.4
    # ── phase gating (the production saturation lesson) ──
    # At production noise levels most rows' true R sits at/above the 0.999 clamp, where
    # NSR = R/(1-R) is pinned at its ceiling: the meter CANNOT SEE those rows -- and
    # during cross-attention renegotiation that is a true statement about the world
    # (follower rows adapt to a moving x-attn; their signal's validity interval is
    # shorter than the corroboration window, so nothing is provable). Rows therefore
    # commit a differentiated lam ONLY when persistently resolvable; saturated rows hold
    # the layer lam. The saturated fraction is printed as the PHASE METER: it should
    # fall, followers-first, when the meaning<->placement handshake closes.
    ROW_SAT_IN = 400.0    # a row starts counting only below this (smoothed NSR)
    ROW_SAT_OUT = 600.0   # ...and keeps its streak until above this: hysteresis, because
                          # a single threshold churns at the boundary (observed live: SAT
                          # ~51% with row-NSR p50 = 679 -> counters reset forever,
                          # eligible pinned at 0). Ceiling is ~999.
    ROW_PERSIST = 3       # consecutive resolvable windows before a row may commit
    ROW_CAP_MULT = 3.0    # row lam <= this x the layer lam (fluke tails burned 50x before)

    def __init__(self, layers, lr, lam0, every, sidecar=None, epoch_updates=0,
                 per_row=False, now_t=0, verbose=True):
        from prototype_packed_2fast import register_nsr_meter
        self.lr = float(lr)
        self.lam0 = float(lam0)
        self.every = max(16, int(every))
        self.per_row = bool(per_row)
        # epoch-revisit guard (see the FLOOR paragraph in the docstring): window <= epoch/4,
        # capped at 1.0 for degenerate tiny epochs; legacy 0.02 when the epoch is unknown
        self.lam_lo = min(4.0 / float(epoch_updates), 1.0) if epoch_updates > 0 else 0.02
        self.C = None                 # telemetry: recomputed from the smoothed median each seed
        self._last_t = -1
        self.sidecar = sidecar
        self._sidecar_warned = False
        self.layers = [m for m in layers if hasattr(m, "packed_w")]
        self.names = [str(getattr(m, "_concord_name", f"#L{i}"))
                      for i, m in enumerate(self.layers)]
        self.nsr_ew = [None] * len(self.layers)
        self._gen = torch.Generator().manual_seed(526071)
        for m in self.layers:
            D = m.packed_w.numel()
            k = min(self.K, D)
            # randint, not randperm: a full D-element permutation per layer (D up to
            # ~6.5M) stalled the seeding boundary for tens of seconds host-side
            # (audit A1); k=128 draws with replacement collide negligibly at
            # production D and the sketch is an unbiased estimator either way
            idx = torch.randint(0, D, (k,), generator=self._gen).to(device=m.packed_w.device)
            buf = torch.zeros(k + 2, dtype=torch.float32, device=m.packed_w.device)
            m._nsr_buf, m._nsr_idx = buf, idx
            register_nsr_meter(m.packed_w, buf, idx)
        self.r_ew = [None] * len(self.layers)          # per-row EW of arm R (device tensors)
        self._row_below = [None] * len(self.layers)    # consecutive-resolvable counters [N]
        self._row_lam_prev = [None] * len(self.layers)  # last committed row kappas (drift meter)
        self._layer_lam = {}                            # layer lam per index (masked-row fallback)
        if self.per_row:
            for m in self.layers:
                N = int(m.packed_w.shape[0])
                buf0 = getattr(m, "_gf_consol_buf", None)
                if buf0 is None or buf0.numel() != N:
                    # PRE-CAPTURE arming: the [N] buffer selects the GF_PER_ROW kernel
                    # variant by numel and its POINTER is baked into any captured graph
                    # -- replace it only here (the seeder builds before first capture),
                    # never after; later commits fill/copy in place.
                    v0 = float(buf0.reshape(-1)[0].item()) if (buf0 is not None
                                                               and buf0.numel() > 0) \
                        else float(getattr(m, "gf_consol", 0.0) or 0.0)
                    m._gf_consol_buf = torch.full((N,), v0, dtype=torch.float32,
                                                  device=m.packed_w.device)
        restored = self._restore(now_t=now_t)
        if verbose:
            print(f"[concord] NOISE-SEED SERVO armed (WHITENED lam=c/NSR"
                  + (", PER-ROW arm meter" if self.per_row else "") + "): "
                  f"{len(self.layers)} layers, sketch k={self.K}, "
                  f"window={self.every} updates; C re-anchors each seed so the median lam stays "
                  f"at the configured {self.lam0:g}"
                  + (f"; {restored} smoothed NSRs restored from sidecar" if restored else ""),
                  flush=True)
        if restored:
            self._commit(tag="restored")   # no flat interlude: seed immediately from the sidecar

    @staticmethod
    def _short(n):
        return (n.replace("down_blocks", "dn").replace("up_blocks", "up")
                 .replace("mid_block", "mid").replace("attentions", "at")
                 .replace("transformer_blocks", "tb").replace(".attn", ".a"))

    def _restore(self, now_t=0):
        if not self.sidecar:
            return 0
        import json
        import os
        try:
            if not os.path.exists(self.sidecar):
                return 0
            with open(self.sidecar, encoding="utf-8") as f:
                d = json.load(f)
            if int(d.get("ver", -1)) != self.SIDECAR_VER:
                return 0
            if int(d.get("step", 0)) > int(now_t) + 5 * self.every:
                # sidecar from a NEWER clock than this run's: a fresh run in an old
                # workspace (the clock resets; the file does not). Restoring would seed
                # a fresh model from another run's converged-state NSRs -- start cold.
                print(f"[concord-nsr] sidecar is from another run lineage (saved t="
                      f"{int(d.get('step', 0))} > current t={int(now_t)}) -> starting cold",
                      flush=True)
                return 0
            byname = d.get("nsr_ew", {})
            n = 0
            for i, nm in enumerate(self.names):
                v = byname.get(nm)
                if v is not None and float(v) > 0.0:
                    self.nsr_ew[i] = float(v)
                    n += 1
            return n
        except Exception as e:
            print(f"[concord-nsr] sidecar unreadable ({type(e).__name__}) -> starting cold",
                  flush=True)
            return 0

    def _save(self, t):
        if not self.sidecar:
            return
        import json
        try:
            with open(self.sidecar, "w", encoding="utf-8") as f:
                json.dump({"ver": self.SIDECAR_VER, "step": int(t), "every": self.every,
                           "nsr_ew": {nm: v for nm, v in zip(self.names, self.nsr_ew)
                                      if v is not None}}, f)
        except Exception as e:
            if not self._sidecar_warned:
                self._sidecar_warned = True
                print(f"[concord-nsr] sidecar write failed ({type(e).__name__}); continuing "
                      f"without persistence", flush=True)

    def _commit(self, t=None, tag=""):
        vals = [(i, v) for i, v in enumerate(self.nsr_ew) if v is not None]
        if not vals:
            return
        svals = sorted(v for _, v in vals)
        med = svals[len(svals) // 2]
        if med <= 0.0:
            return
        # re-anchor each seed: the median layer runs exactly the configured lam
        # (whitened law: lam = (lam0*med)/NSR)
        self.C = self.lam0 * med
        lams = []
        n_floor = 0
        for i, v in vals:
            raw = self.C / max(v, 1e-9)
            lam = min(max(raw, self.lam_lo), 1.5)   # ceiling/floor: see the docstring clamp paragraph
            n_floor += 1 if raw < self.lam_lo else 0
            self.layers[i].gf_consol = lam / max(self.lr, 1e-12)   # property setter fills the device buf
            self._layer_lam[i] = lam                # masked-row fallback for the per-row commit
            lams.append(lam)
        lams.sort()
        hi_i, hi_v = max(vals, key=lambda iv: iv[1])
        lo_i, lo_v = min(vals, key=lambda iv: iv[1])
        print(f"[concord-nsr] {tag if tag else f't={t}'}: seeded {len(lams)}/{len(self.layers)} | "
              f"NSR~ min/med/max={svals[0]:.1f}/{med:.1f}/{svals[-1]:.1f} | "
              f"lam min/med/max={lams[0]:.3f}/{lams[len(lams) // 2]:.3f}/{lams[-1]:.3f} "
              f"(C={self.C:.3g}, floor={self.lam_lo:.4g} clipped {n_floor}) | "
              f"hot {self._short(self.names[hi_i])}={hi_v:.0f} "
              f"cold {self._short(self.names[lo_i])}={lo_v:.1f}", flush=True)

    @staticmethod
    @torch.no_grad()
    def _row_arm_stats(m):
        """Per-row arm stats from the packed word: R = sum(gap_w^2)/sum(sum_w^2) and mean
        arm occupancy in int units. Row exponents cancel in the ratio; COLUMN exponents do
        not and are applied (the -15 mantissa bias cancels too)."""
        p = m.packed_w.to(torch.int32)
        eL = (p >> 24).float()
        eH = ((p << 8) >> 24).float()
        w = torch.pow(2.0, m.arm_col_exp.float() + m.col_exp.float())[None, :]
        gap2 = ((eL - eH) * w).pow(2).sum(dim=1)
        sum2 = ((eL + eH) * w).pow(2).sum(dim=1)
        occ = (eL.abs() + eH.abs()).mean(dim=1) * 0.5
        return (gap2 / sum2.clamp_min(1e-30)).clamp(1e-4, 0.999), occ

    @torch.no_grad()
    def _commit_rows(self, t):
        """Per-row whitened commit from the EW-smoothed arm R (see the PER-ROW block in
        the class constants for the rationale, the closed-loop caveat, and the exp56
        gate). Runs AFTER _commit: the layer fill is already in the buffers, so masked
        (unreadable) rows simply keep it."""
        stats = []
        for i, m in enumerate(self.layers):
            R, occ = self._row_arm_stats(m)
            prev = self.r_ew[i]
            self.r_ew[i] = R if prev is None else \
                (1.0 - self.ROW_EW_BETA) * prev + self.ROW_EW_BETA * R
            stats.append((self.r_ew[i], occ))
        readable = [(r / (1.0 - r))[occ >= self.ROW_OCC_MIN] for r, occ in stats]
        cat = torch.cat([x for x in readable if x.numel() > 0]) \
            if any(x.numel() > 0 for x in readable) else None
        if cat is None or cat.numel() < 100:
            # residue phase: the LAYER lams must be authoritative -- stale row kappas
            # left on the modules would keep steering the ramp/mod writers forever
            for m in self.layers:
                if getattr(m, "_gf_consol_rows", None) is not None:
                    m._gf_consol_rows = None
            return
        med = float(cat.median())
        c = self.lam0 * max(med, 1e-6)      # global row-median anchor (whitened law)
        drifts = []
        n_rows = n_masked = n_sat = n_eligible = 0
        for i, m in enumerate(self.layers):
            r, occ = stats[i]
            nsr = (r / (1.0 - r)).clamp_min(1e-6)
            mask = occ >= self.ROW_OCC_MIN
            # phase gate: only rows PERSISTENTLY below saturation may differentiate
            _th = torch.where(cnt_prev > 0, self.ROW_SAT_OUT, self.ROW_SAT_IN)                 if (cnt_prev := self._row_below[i]) is not None                 else self.ROW_SAT_IN
            below = (nsr < _th) & mask
            cnt = self._row_below[i]
            if cnt is None:
                cnt = torch.zeros_like(occ, dtype=torch.int16)
            cnt = torch.where(below, (cnt + 1).clamp(max=30000),
                              torch.zeros_like(cnt))
            self._row_below[i] = cnt
            eligible = mask & (cnt >= self.ROW_PERSIST)
            layer_lam = float(self._layer_lam.get(i, self.lam0))
            cap = min(1.5, self.ROW_CAP_MULT * layer_lam)
            lam_rows = (c / nsr).clamp(self.lam_lo, 1.5).clamp(max=cap)
            base = layer_lam / max(self.lr, 1e-12)
            kap = torch.where(eligible, lam_rows / max(self.lr, 1e-12),
                              torch.full_like(lam_rows, base))
            n_sat += int((mask & (nsr >= self.ROW_SAT_OUT)).sum())
            n_eligible += int(eligible.sum())
            prev_k = self._row_lam_prev[i]
            if prev_k is not None and bool(eligible.any()):
                d = ((kap - prev_k).abs() / prev_k.clamp_min(1e-12))[eligible]
                drifts.append(float(d.median()))
            self._row_lam_prev[i] = kap.clone()
            buf = m._gf_consol_buf
            if buf.numel() == kap.numel():
                # UNRAMPED row kappas live on the module: the per-step
                # dissipation_fill_ramp writer re-derives the buffer from these each
                # step (a scalar fill_ would flatten the row structure -- the row law
                # was ~100% inert until that writer went row-aware; audit C-4)
                m._gf_consol_rows = kap.clone()
                buf.copy_(kap)
            n_rows += int(mask.numel())
            n_masked += int((~mask).sum())
        qs = torch.quantile(cat, torch.tensor([0.1, 0.5, 0.9], device=cat.device))
        drift = (f"{sorted(drifts)[len(drifts) // 2]:.3f}" if drifts
                 else "n/a(0 eligible)")
        _readable = n_rows - n_masked
        print(f"[concord-nsr-rows] t={t}: {_readable}/{n_rows} rows readable "
              f"({n_masked} residue-masked) | SAT={100.0 * n_sat / max(_readable, 1):.1f}% "
              f"of readable (phase meter: falls when the x-attn handshake closes) | "
              f"eligible={n_eligible} (persist>={self.ROW_PERSIST}, cap {self.ROW_CAP_MULT}x layer) | "
              f"NSR~row p10/50/90={float(qs[0]):.0f}/{float(qs[1]):.0f}/{float(qs[2]):.0f} | "
              f"drift p50={drift}", flush=True)

    @torch.no_grad()
    def step(self, t):
        if t <= 0 or (t % self.every) != 0 or t == self._last_t:
            return
        self._last_t = t
        harvested = 0
        for i, m in enumerate(self.layers):
            buf, idx = m._nsr_buf, m._nsr_idx
            k = idx.numel()
            T = float(buf[k])
            if T < self.every * 0.5:        # under-filled window (resume/recapture boundary)
                buf.zero_()
                continue
            D = m.packed_w.numel()
            mean_sk = buf[:k] / T
            eg2 = float(mean_sk.pow(2).sum()) * (D / k)   # ||E g||^2 estimate
            g2 = float(buf[k + 1]) / T                    # E ||g||^2 per micro
            buf.zero_()
            # redraw the sketch for the NEXT window, IN PLACE: the captured-graph gather
            # reads this tensor by pointer, so new coords must arrive under the same
            # storage (device tensors cross the graph boundary; python objects don't)
            nidx = torch.randint(0, D, (k,), generator=self._gen).to(device=idx.device,
                                                               dtype=idx.dtype)
            idx.copy_(nidx)
            if eg2 <= 0.0 or g2 <= 0.0:
                continue
            nsr = max(g2 / eg2 - 1.0, 1e-3)
            prev = self.nsr_ew[i]
            self.nsr_ew[i] = nsr if prev is None else \
                (1.0 - self.EW_BETA) * prev + self.EW_BETA * nsr
            harvested += 1
        if harvested == 0:
            return
        self._commit(t=t)
        self._save(t)
        if self.per_row:
            self._commit_rows(t)


class ConcordController:
    """Holds the swapped Concord UNet layers + the per-step schedule + the rebalance gate
    for one training run. Created in the SDXL setup (after the model is loaded, before the
    optimizer is built); driven by the trainer via before_step()/after_step()."""

    def __init__(self, unet, device, learning_rate: float, total_steps: int, optimizer_config=None,
                 module_filters=None, text_encoder=None, te_lr=None, te_wd_anchor=0.5,
                 text_encoder_2=None, te2_lr=None, te_chase_alpha=None,
                 te_use_anchor=True, te2_use_anchor=True, workspace_dir=None):
        from concord_winner import swap_unet_to_winner, GatedRebalance, swap_text_encoder_to_anchor, \
            swap_text_encoder_to_winner, \
            set_lazy_gate, set_lazy_thresh, set_min_leak, set_evap_build_min, set_lamb_trust, \
            set_coh_vhat, set_coh_kappa, set_evap_slack
        self.config = make_concord_config(learning_rate, optimizer_config)
        # emb seeder runs on the row EVIDENCE clock (exp55) -- rates set at each seeding
        self._enr_rate = {}    # per-plane [K] sighting rates (per update)
        # workspace dir for controller-owned sidecars (the NSR sidecar must NOT depend on
        # the EMB calib path: without registered embeddings that path is None and the UNet
        # seeder's persistence silently dies)
        self.workspace_dir = str(workspace_dir) if workspace_dir else None
        # D3 guard: step_cap and gf_trust_delta_sq are the two step bounds (hard clamp vs the
        # rank-1-Adam denominator). With BOTH <= 0 the denom collapses to eps and step_cap=0
        # zeroes every step -> the UNet never learns. Unexposed knobs that serialize to 0.0 are
        # honored verbatim by pick(), so this pair can arise silently. Restore the v_hat denom
        # rather than train with a dead step; paired with the kernel's step_cap<=0 "no clamp"
        # guard, step_cap-off then runs as bounded rank-1 Adam.
        if float(self.config.step_cap) <= 0.0 and float(self.config.gf_trust_delta_sq) <= 0.0:
            self.config.gf_trust_delta_sq = 1.0
            print("[concord] guard: step_cap and gf_trust_delta_sq both <=0 (would zero the "
                  "step) -> restored gf_trust_delta_sq=1.0 (v_hat IS the denom)", flush=True)
        # Dimensionless dissipation: the physical friction knob is lam = lr*kappa
        # (u <- u - lr*kappa*(1-coh)*u). When `dissipation` is set it overrides
        # gf_consol with lam/lr, so the same lam means the same per-step friction
        # at ANY learning rate (kappa alone does not transfer: kappa=50 at SDXL
        # lr 7.5e-5 is lam=0.00375 — ~100x under the CPU noisy-regime optimum).
        # The lr*kappa < 2 stability guard then reads directly as lam < 2.
        if self.config.dissipation is not None:
            lam = float(self.config.dissipation)
            self.config.gf_consol = lam / max(self.config.lr, 1e-12)
            print(f"[concord] dimensionless dissipation lam={lam:g} @ lr={self.config.lr:g} "
                  f"-> gf_consol={self.config.gf_consol:.0f}")
        self.total_steps = max(1, int(total_steps))
        # module_filters: OneTrainer's layer_filter (ModuleFilter list). When set to a non-"full"
        # preset (e.g. attn-mlp -> ["attentions"]) only the selected layers are swapped to
        # Concord; the rest stay standard bf16 and are frozen, dropping their packed state.
        # unet is None when config.unet.train is False -> SKIP the swap entirely: the UNet stays
        # a standard FROZEN module (no packed state, no per-step self-step), mirroring the TE
        # (only swapped when text_encoder.train). The backward still flows through the frozen UNet
        # to reach the trainable embeddings. self.layers==[] is a supported state -- GatedRebalance,
        # the servo, and every before/after_step loop guard or loop-tolerate an empty layer set.
        # 2-fast startup self-test: the 2-fast kernels compile at first
        # launch; run one tiny layer BEFORE any UNet surgery so a compile
        # or sanity failure aborts loudly at t=0 instead of mid-run.
        from prototype_packed_b import (set_evict_valve, set_evict_gain, set_evict_cf_gate,
                                        set_heldout_router, set_router_noise,
                                        set_perfcoh)
        set_evict_valve(bool(getattr(self.config, "evict_valve", False)))
        _cf_gate = bool(getattr(self.config, "evict_cf_gate", False))
        set_evict_cf_gate(_cf_gate)
        # Held-out arm router (2-fast). The accum>=2 guard lives in the SDXL setup (which
        # knows gradient_accumulation_steps) and force-clears the module flag; both it and
        # this set run before the first kernel launch, where the constexpr actually bakes.
        _rt = bool(getattr(self.config, "heldout_router", False))
        set_heldout_router(_rt)
        set_router_noise(_rt and bool(getattr(self.config, "router_coh_noise", False)))
        _pc = bool(getattr(self.config, "perfcoh_partition", False))
        set_perfcoh(_pc, float(getattr(self.config, "perfcoh_tau", 0.5) or 0.5))
        if _pc:
            print("[concord] PERFCOH PARTITION ON (soft CF gate): anchor commits *= exp(-(1-coh)/tau), "
                  f"eviction reverts *= the complement (tau={float(getattr(self.config, 'perfcoh_tau', 0.5) or 0.5):g}). "
                  "Marginal admissions stay on probation instead of ratcheting into the reference. "
                  "Baked at capture.", flush=True)
        if _rt:
            print("[concord] HELD-OUT ARM ROUTER ON (2-fast): alternate micros -> alternate arms; "
                  "gap = cross-split disagreement"
                  + (", fed into coherence noise (un-discountable floor, exp47b)"
                     if bool(getattr(self.config, "router_coh_noise", False)) else "")
                  + ". Baked at capture.", flush=True)
            if bool(getattr(self.config, "noise", False)):
                print("[concord] WARNING: fluctuation noise + heldout_router -- injected noise routes "
                      "wholly to the on-duty arm each micro (matched-pair gap attribution does not "
                      "hold); the gap carries an extra zero-mean noise source. Uncharacterized.", flush=True)
        # cf-gate is a high-gain enabler (exp 26f): a net loss at low gain, a
        # frontier gain only near rate 1.0 -> LOCK the gain to 1.0 when on.
        _eg = 1.0 if _cf_gate else float(getattr(self.config, "evict_gain", 0.66) or 0.66)
        set_evict_gain(_eg)
        if bool(getattr(self.config, "evict_valve", False)):
            print(f"[concord] EVICTION VALVE ON (exp 26, delta-gated @ gain {_eg:g}"
                  f"{', cf-gated (gain locked 1.0)' if _cf_gate else ''}): "
                  "demotes on sign disagreement with the LEARNED DELTA (s_slow-v_slow), "
                  "NOT raw position -- the pretrained prior (common mode) is protected "
                  "while learned cruft the gradient reverses is drained. BOTH formats. "
                  "Expect ||deploy|| to plateau. Kill switch: optimizer option "
                  "evict_valve=False.", flush=True)
        if bool(getattr(self.config, "two_fast", False)) and unet is not None:
            from prototype_packed_2fast import two_fast_self_test
            two_fast_self_test(device)
        self.layers = swap_unet_to_winner(
            unet, device, self.config.lr, gf_consol=self.config.gf_consol,
            step_cap=self.config.step_cap, gf_trust_delta_sq=self.config.gf_trust_delta_sq,
            verbose=False, module_filters=module_filters,
            train_cond_embed=bool(getattr(self.config, "concord_train_cond_embed", False)),
            conv_full_vhat=bool(getattr(self.config, "concord_conv_full_vhat", False)),
            kaiming_init=bool(self.config.kaiming_init),
            kaiming_scale=float(self.config.kaiming_scale),
            two_fast=bool(getattr(self.config, "two_fast", False)),
            bracket_d=float(getattr(self.config, "bracket_d", 0.25))) if unet is not None else []
        self.gate = GatedRebalance(self.layers)
        # Arm-plane ratchet gate (2-fast only): the word-level GatedRebalance
        # fires ~never at finetune lr, but the ARM exponents must still adapt
        # -- their own gated dispatcher runs beside it (urgent up-ticks on a
        # shared-reduction trigger, lazy down-ticks on a fixed cadence).
        self.arm_gate = None
        if any(hasattr(m, "arm_ratchet") for m in self.layers):
            from prototype_packed_2fast import GatedArmRatchet
            self.arm_gate = GatedArmRatchet(self.layers)
            # Prequential servo meter (exp 24): per-layer <g, A_gap>
            # accumulated in-kernel on every launch. PASSIVE for now --
            # _log_health prints the window cosine; nothing actuates on
            # it. Registered before any capture; the eviction restore
            # re-registers. CONCORD_PREQ_METER=0 disables (and the kernel
            # branch compiles out).
            import os as _po
            if _po.environ.get("CONCORD_PREQ_METER", "1") != "0":
                from prototype_packed_2fast import register_preq_meter
                for _m in self.layers:
                    if hasattr(_m, "arm_ratchet"):
                        _m._preq_meter = torch.zeros(
                            4, dtype=torch.float32,   # [0]=sum g*gap [1]=sum g^2 [2]=sum gap^2 [3]=sum|g*gap| (gross)
                            device=_m.packed_w.device)
                        register_preq_meter(_m.packed_w, _m._preq_meter)
            # Graph-native gradient-SNR sketch (CSNR meter). ON via env CONCORD_CSNR_METER=1 (or
            # config.concord_csnr_meter). Registered PRE-capture so the kernel bakes the per-layer
            # sketch buffers; when off nothing is registered and _sketch_gy_kernel never launches
            # (bit-identical). Collector built lazily on the first on_timesteps. Env-gated so an
            # UNVALIDATED sketch kernel can be backed out by unsetting the var + relaunch.
            import os as _po
            self._csnr = None; self._csnr_bufs = []; self._csnr_pool = []
            self._csnr_last_arm = -10 ** 9; self._csnr_curve = None
            _csnr_on = (bool(getattr(self.config, "concord_csnr_meter", False))
                        or _po.environ.get("CONCORD_CSNR_METER", "").strip().lower() in ("1", "true", "yes", "on"))
            if _csnr_on and self.layers:
                from prototype_packed_2fast import register_sketch_meter
                _sk = self.layers[::max(1, len(self.layers) // 8)][:8]
                _keach = max(1, 256 // max(1, len(_sk)))
                for _i, _m in enumerate(_sk):
                    self._csnr_bufs.append(register_sketch_meter(_m.packed_w, _keach, 1234 + _i))
                # accum = gradient_accumulation_steps (pool that many micro-batches per observation).
                # The ConcordConfig does not carry it, so read CONCORD_CSNR_ACCUM (default 1 -> SET it
                # to your gradient_accumulation_steps, else the meter runs in the sparse per-micro-batch
                # regime that the CPU de-risk found marginal).
                self._csnr_accum = max(1, int(_po.environ.get("CONCORD_CSNR_ACCUM", "1")))
                self._csnr_window = int(_po.environ.get("CONCORD_CSNR_WINDOW", "150"))
                print(f"[concord] CSNR meter ON (collect+log; sampler is NOT consuming the curve yet): "
                      f"{len(_sk)} layers x {_keach} coords, window={self._csnr_window} steps, "
                      f"accum={self._csnr_accum} (set CONCORD_CSNR_ACCUM to your grad-accum steps).",
                      flush=True)
        # Frozen-anchor TE training: CLIP-L (TE1) and/or CLIP-G (TE2), each anchored with its OWN
        # lr; swapped AFTER the UNet so the shared global coh flags are already set. te_groups
        # carries (layers, lr) per encoder for the schedule; te_layers is the combined list for the
        # deploy-bridge filter; te_encoders the live modules that bridge iterates. Each empty unless
        # its encoder is passed (TE2 only when the caller opts in via concord_te2_anchor).
        self.te_lr = float(te_lr) if te_lr else self.config.lr
        self.te2_lr = float(te2_lr) if te2_lr else self.te_lr
        # ANCHOR-PATH config below (only consulted when te_use_anchor=True, i.e. concord_te_anchor /
        # concord_te2_anchor = True). The TOP-LEVEL TE default is now the WINNER recipe (train like
        # the UNet -- see _swap_te / swap_text_encoder_to_winner); the anchor is opt-in. WITHIN the
        # anchor path it defaults to FROZEN (creep binned 2026-06-18): v_slow pinned at pretrained W
        # (alpha_v_fast=0), gate inert (C*=0), wd_anchor decays the delta (s_slow,s_fast) toward 0 --
        # which, because v_slow is pinned at W, is a SYMMETRIC restore of the live weight toward
        # pretrained. The validated low-drift mode; every toggle in it is self-consistent.
        # The CREEP variant (a tiny alpha_v_fast lets v_slow creep so the gate goes live) is retained
        # OPT-IN ONLY: set CONCORD_TE_CREEP_ALPHA>0. NOT recommended — its "anchor" creeps off
        # pretrained (no fixed reference) and wd_anchor there decays toward ZERO, not pretrained (the
        # mis-semantic the anchor audit flagged). CONCORD_TE_CREEP_GF is unchanged (creep-only).
        import os as _os
        _cav = _os.environ.get("CONCORD_TE_CREEP_ALPHA")
        if _cav is None or not _cav.strip():
            _creep_av = None                        # default: FROZEN anchor (creep binned)
        else:
            try:
                _v = float(_cav)
            except ValueError:
                _v = -1.0                           # "off"/"frozen"/garbage -> opt out
            _creep_av = _v if _v > 0.0 else None    # <=0 -> frozen anchor; >0 -> opt-in creep
        _creep_gf = float(_os.environ.get("CONCORD_TE_CREEP_GF", "0") or 0.0)
        # TE swap mode, PER ENCODER: WINNER recipe (default = train like the UNet) or the frozen
        # ANCHOR (opt-in via te_use_anchor/te2_use_anchor, bound by the setup to concord_te_anchor/
        # concord_te2_anchor). Winner shares the UNet's gf_consol/step_cap/gf_trust so the TE runs
        # the same dissipation/step bounds; the anchor path keeps the creep/wd_anchor knobs.
        # Off-by-default: extend the per-row dynamic-lambda servo to the WINNER TEs by swapping them
        # 2-fast (so the ARM meter can read their held-out e_L/e_H arms). Master-gated by
        # CONCORD_TE_ROW_SERVO (env/sentinel) AND the run already being two_fast+heldout_router (the
        # router is what makes the arm gap a cross-split data meter). OFF => winner TEs swap single-
        # fast exactly as today (bit-identical layout/deploy/trajectory). Read ONCE here, pre-swap.
        from modules.util.optimizer.concord_graph import te_row_servo_opted_in as _te_srv_optin
        self._te_row_servo = (_te_srv_optin()
                              and bool(getattr(self.config, "two_fast", False))
                              and bool(getattr(self.config, "heldout_router", False))
                              and bool(getattr(self.config, "noise_seed_servo", False)))
        if _te_srv_optin() and not self._te_row_servo:
            print("[concord] CONCORD_TE_ROW_SERVO set but two_fast/heldout_router/noise_seed_servo "
                  "are not all ON -> winner TEs stay single-fast (the per-row arm servo needs the "
                  "2-fast router + the seeder). No-op.", flush=True)
        if self._te_row_servo:
            print("[concord] TE-ROW-SERVO ON: winner TEs -> 2-fast (held-out arms); per-row dynamic "
                  "lambda extended to the text encoders. UNVALIDATED on TEs -- A/B the DEPLOYED "
                  "samples AND run the memorization/label-noise check before trusting it.", flush=True)
        def _swap_te(_te, _lr, _use_anchor):
            if _te is None:
                return []
            if _use_anchor:
                return swap_text_encoder_to_anchor(_te, device, _lr, te_wd_anchor,
                                                   alpha=te_chase_alpha,
                                                   creep_alpha_v=_creep_av, creep_gf_consol=_creep_gf)
            return swap_text_encoder_to_winner(_te, device, _lr,
                                               gf_consol=self.config.gf_consol,
                                               step_cap=self.config.step_cap,
                                               gf_trust_delta_sq=self.config.gf_trust_delta_sq,
                                               two_fast=self._te_row_servo,   # single-fast unless opted in
                                               bracket_d=float(getattr(self.config, "bracket_d", 0.25)),
                                               verbose=True)   # loud per-encoder mode log at startup
        _te1 = _swap_te(text_encoder, self.te_lr, te_use_anchor)
        _te2 = _swap_te(text_encoder_2, self.te2_lr, te2_use_anchor)
        self.text_encoder = text_encoder          # TE1 (back-compat ref)
        self.te_encoders = [te for te in (text_encoder, text_encoder_2) if te is not None]
        self.te_groups = [(lyr, lr) for lyr, lr in ((_te1, self.te_lr), (_te2, self.te2_lr)) if lyr]
        self.te_layers = _te1 + _te2              # combined: deploy-bridge filter + "any TE?" check
        self.te_gates = [GatedRebalance(lyr) for lyr, _lr in self.te_groups]
        # Winner-recipe TEs have drift_cancel_C>0 and wd_anchor=0, so the kernel's write_boil gate is
        # TRUE -- they would atomic-add into the SHARED per-device _boil_buf/_memgap_buf that
        # read_flow_audit reports as the UNET audit (the [loss]-line boil/waste). Route their meters
        # to a TE-only scratch sink so the UNet audit stays UNet-only. Registered BEFORE graph capture
        # (graph-safe baked pointer), reusing the per-layer routing the servo uses. Frozen anchors
        # (drift_cancel_C=0) never write boil, so this touches winner TEs only.
        _winner_te = [m for m in self.te_layers if getattr(m, "alpha_v_fast", 0.0) > 0.0]
        if _winner_te:
            from prototype_packed_b import register_layer_meters
            _dev = _winner_te[0].packed_w.device
            self._te_boil_scratch = torch.zeros(6, dtype=torch.float32, device=_dev)   # [0..3] boil/waste; [4],[5] = M6a diversity meter
            self._te_memgap_scratch = torch.zeros(1, dtype=torch.float32, device=_dev)
            for _m in _winner_te:
                register_layer_meters(_m.packed_w, self._te_boil_scratch, self._te_memgap_scratch)
        # TE-row-servo (self._te_row_servo): when the winner TEs are 2-fast, give them the same
        # arm-plane ratchet + prequential meter the UNet 2-fast layers get (self.arm_gate/preq at
        # ~612-630 is built over self.layers BEFORE the TE swap, so it cannot include TEs). PRE-CAPTURE
        # (still in __init__). Natural no-op when the TEs are single-fast: GatedArmRatchet self-filters
        # on hasattr(arm_ratchet), which single-fast layers lack, so _te_2fast is empty.
        self.te_arm_gate = None
        _te_2fast = [m for m in _winner_te if hasattr(m, "arm_ratchet")]
        if _te_2fast:
            from prototype_packed_2fast import GatedArmRatchet
            self.te_arm_gate = GatedArmRatchet(_te_2fast)
            import os as _po_te
            if _po_te.environ.get("CONCORD_PREQ_METER", "1") != "0":
                from prototype_packed_2fast import register_preq_meter
                for _m in _te_2fast:
                    _m._preq_meter = torch.zeros(4, dtype=torch.float32, device=_m.packed_w.device)
                    register_preq_meter(_m.packed_w, _m._preq_meter)
        # Lazy-update gate is a module-level global read at every kernel launch; swap_unet_to_winner
        # forces the coherence/noise flags but not this one, so set it explicitly from config here.
        set_lazy_gate(self.config.lazy_gate)
        set_lazy_thresh(self.config.lazy_active_thresh)
        # Servo min-leak floor: module global like the lazy gate (per-run constant,
        # baked at CUDA-graph capture). Guards the lam -> 1 regime from slam-shut.
        set_min_leak(self.config.min_leak)
        # Hypothesis-infancy guard: no dissipation below one deploy tick.
        set_evap_build_min(self.config.evap_build_min)
        # Per-layer LAMB relative-step trust cap: brakes the small-norm (texture-conv) overcook.
        # Module global like min_leak; CUDA-graph safe (static per-layer buffer, device-only recompute).
        set_lamb_trust(self.config.lamb_trust, self.config.lamb_cap, self.config.lamb_clip)
        # cf-modulated coherence: discount the Wiener residual by the coherent fraction cf=d_sv^2/v_hat
        # so common-concept diversity isn't dissipated as noise (low-cf noise still killed; knee=coh_kappa).
        set_coh_vhat(self.config.coh_vhat)
        set_coh_kappa(self.config.coh_kappa)
        set_evap_slack(float(getattr(self.config, "concord_evap_slack", 0.25)))   # EVAP clamp: kill on min(coh, coh_raw+slack)
        # Dissipation autotuner (probe-then-commit), opt-in via optimizer.autotune_table.
        # Built LAZILY on the first before_step(): total_steps here is a placeholder —
        # the trainer finalizes the horizon at train start.
        # The hunting per-layer servos are removed; the NoiseScaleSeeder (noise_seed_servo)
        # is the only dissipation controller. self.autotuner is only ever the table tuner
        # (DissipationAutoTuner) or None; self.autotuners stays permanently [] -- both kept
        # because external readers probe them defensively.
        self.autotuner = None
        self.autotuners = []
        self._autotune_pending = bool(getattr(self.config, "autotune_table", None))
        # common-mode meter (log-only, 1x/epoch): top-k SVD of each emb core's epoch-increment
        # accumulator -> how much of every token's step lives in the shared subspace, and how
        # stable that subspace is epoch-over-epoch. Pure diagnostic for the deflation-shield
        # design; never touches the update path. Opt-out: CONCORD_COMMONMODE_METER=0.
        self._cm_prev = {}          # id(core) -> (accum copy at last boundary, top-k V [dim,k])
        self._cm_last_t = -1
        # deflation gate state (armed by the meter when concord_emb_deflate is on; cores reset
        # every exit-42 segment, so arming persists via the concord_commonmode.json sidecar with
        # owners stored by token NAME -- row order is not segment-stable).
        self.emb_deflate = False
        self.emb_deflate_gamma = 0.5
        self.emb_deflate_modes = 8   # cap on armed common-mode components per epoch (<= K_TOP)
        self._cm_sidecar = None
        self._cm_side_prev = {}     # core idx -> prev V [dim,k] loaded from sidecar (stability across segments)
        self._cm_state = {}         # core idx -> armed component list (for the sidecar)
        self._snr_mod_announced = False
        self._current_fill_ramp = 1.0
        self._last_boil_protected = None   # cf-discounted boil, set by read_flow_audit for the TB log
        self._last_m6a = None   # M6a dissipation-space diversity meter ([4]/[5]), set by read_flow_audit (log-only)
        self.emb_cores = []         # packed-embedding cores (register_embedding_cores)
        self.emb_trainables = []    # the ConcordPackedEmbedding modules (drive/accum)
        self.emb_row_names = []     # row -> placeholder (calibration report)
        self.emb_lr = 0.0
        self.emb_delay_epochs = 0.0  # "divot" (register_embedding_cores sets it)
        self.emb_delay_steps = 0     # resolved at horizon finalize (steps/epoch known)
        self.emb_auto_drive = False  # normalize per-token drive from the divot window
        self.emb_freq_exponent = 0.5  # beta in drive = (med(n)/n)^beta
        self.emb_calib_path = None   # workspace sidecar (counts = dataset property)
        self.emb_window_report = False  # per-token Wiener posterior around init (diagnostic)
        self._emb_drive_applied = False
        self.steps_per_epoch = 0.0
        self.step_idx = 0
        print(f"[concord] swapped {len(self.layers)} UNet layers | lr={self.config.lr} "
              f"gf_consol={self.config.gf_consol} noise={self.config.noise} "
              f"lazy_gate={self.config.lazy_gate}@{self.config.lazy_active_thresh} "
              f"(horizon set at train start)")

    def _build_autotuner(self):
        """Deferred autotuner construction: needs the FINAL total_steps (the trainer
        finalizes the horizon at train start, after __init__). UNet layers ONLY — the
        frozen-anchor TE runs alpha_v_fast=0, its telescope never advances and its
        coherence reads ~0 regardless of data quality; including it would drag the
        probe mean toward "maximum noise"."""
        import json
        from prototype_packed_b import DissipationAutoTuner
        self._autotune_pending = False
        table = [(float(c), float(k)) for c, k in json.loads(self.config.autotune_table)]
        if self.config.dissipation is not None:
            # dimensionless mode: the table's kappa column is lam = lr*kappa ->
            # convert to kappa at this run's lr BEFORE the stability guard, so the
            # guard's lr*kappa reads exactly the table's lam. The coherence column
            # is untouched — its scale remains domain-calibrated (the exp-11
            # meter-conditioning rule).
            table = [(c, k / max(self.config.lr, 1e-12)) for c, k in table]
        max_kappa = max(k for _, k in table)
        if self.config.lr * max_kappa >= 2.0:
            raise ValueError(
                f"autotune_table max kappa {max_kappa:g} at lr {self.config.lr:g}: "
                f"lr*kappa = {self.config.lr * max_kappa:.2f} >= 2 is linearly unstable "
                f"(u <- u - lr*k*(1-coh)*u). Lower the table's kappa ceiling or the lr.")
        if self.config.gf_consol <= 0:
            raise ValueError(
                "autotune requires gf_consol > 0 (the probe kappa): the kernel's "
                "consolidation branch is baked at CUDA-graph capture from the "
                "capture-time value; gf_consol=0 would bake it OUT and the committed "
                "kappa would be ignored under replay.")
        probe_start = int(0.04 * self.total_steps)
        probe_end = max(int(0.10 * self.total_steps), probe_start + 1)
        # Probe placement (the calibration doc's hard caveat): the window must clear
        # warmup AND the ~1/alpha init-consolidation transient, or the meter reads ~0
        # regardless of data quality and commits the table's max-friction kappa.
        # Observed on a short bs=2 overfit: probe steps 4-12 -> coh 0.002 -> kappa
        # ceiling on CLEAN data (visible deploy damage). Warn-and-misfire was the
        # old behavior; now AUTO-DEFER the window past the transient, and if a
        # clean probe can't fit in the first half of the run, disable the tuner
        # for this run instead of committing garbage (the configured base
        # kappa/dissipation then holds end-to-end).
        transient = int(2.0 / max(self.config.alpha, 1e-6))
        # The meter's "signal" is C*(S - A), and the anchor fills at the leak
        # rate 2*alpha_v_fast (time constant ~500 steps at the winner 0.001).
        # Before ~1 time constant, d_sv is dominated by UN-LEAKED INIT WEIGHT,
        # not learned drift — the probe then reads init residue (~0.5 coh on
        # SDXL, observed) regardless of data quality. A probe is only
        # data-calibrated once the telescope has relaxed.
        telescope = int(0.5 / max(self.config.alpha_v_fast, 1e-9))
        min_start = max(int(self.config.warmup), transient, telescope)
        if probe_start < min_start:
            window = probe_end - probe_start
            probe_start = min_start
            probe_end = probe_start + window
            if probe_end > self.total_steps // 2:
                print(f"[concord] autotune DISABLED for this run: a clean probe "
                      f"window ({window} steps past warmup={self.config.warmup}, "
                      f"the ~{transient}-step init transient, and the "
                      f"~{telescope}-step telescope relaxation) does not fit in "
                      f"the first half of {self.total_steps} steps. The "
                      f"configured kappa (gf_consol={self.config.gf_consol:.0f}) "
                      f"holds end-to-end. Lengthen the run to re-enable "
                      f"autotuning.")
                self.autotuner = None
                return
            print(f"[concord] autotune probe deferred to [{probe_start},{probe_end}) "
                  f"to clear warmup ({self.config.warmup}) / the ~{transient}-step "
                  f"init transient / the ~{telescope}-step telescope relaxation "
                  f"(the meter reads init residue, not data, before they pass).")
        self.autotuner = DissipationAutoTuner(
            self.layers,
            probe_start=probe_start,
            probe_end=probe_end,
            table=table,
            probe_kappa=self.config.gf_consol,
            beta1_on=self.config.autotune_beta1_on,
            beta1_coh_threshold=self.config.autotune_beta1_coh,
            reprobe_band=self.config.autotune_reprobe_band,
            # arm the watchdog only after the telescope has fully settled
            # (~3 time constants) -- before that the meter falls secularly
            # and every windowed mean reads as a "drop"
            watchdog_min_t=3 * telescope)

    # gamma-SNR modulation cap: lam_t = lr*kappa_t never exceeds this (half the
    # lam < 2 linear-stability ceiling), whatever the batch's SNR draw.
    _LAM_MOD_CAP = 1.0

    @staticmethod
    def _fill_ramp(t, alpha_v_fast):
        """Telescope anchor-fill fraction 1 - exp(-2*alpha_v_fast*t): the run-level
        infancy ramp. Friction engages in proportion to how much weight has been
        decided into the anchor -- which is also exactly how data-calibrated the
        coherence meter is (its signal C*(S-A) is init-residue-dominated before
        the anchor fills). alpha_v_fast <= 0 (pinned anchor) => no ramp."""
        if alpha_v_fast <= 0:
            return 1.0
        import math
        return 1.0 - math.exp(-2.0 * alpha_v_fast * t)

    @staticmethod
    def _emb_clock(step_idx, total_steps, delay_steps):
        """Embedding-group schedule clock under the release delay ("divot").
        Returns None while frozen, else (effective_step, effective_total): a
        shifted clock so the released group gets a fresh warmup and its cosine
        still ends at the run horizon. Pure function of step_idx -> resume-safe
        (step_idx is seeded from global_step at horizon finalize)."""
        d = max(0, int(delay_steps))
        if d == 0:
            return step_idx, total_steps
        if step_idx < d:
            return None
        return step_idx - d, max(1, total_steps - d)

    def register_embedding_cores(self, planes, emb_lr, delay_epochs=0.0, noise_seed=False,
                                 auto_drive=False, freq_exponent=0.5,
                                 calib_path=None, window_report=False,
                                 deflate=False, deflate_gamma=0.5, deflate_modes=8):
        """Bring the packed-embedding cores under the controller's physics. As
        created they are the LEAST protected, HIGHEST leverage parameters in
        the system: gf_consol = 0 (zero dissipation -- the only trainables
        without friction), a CONSTANT lr (outside winner_step: no warmup, no
        cosine -- they keep churning at full rate into a converged model, the
        textbook fried-embedding mechanism), per-step norm-pin requantization,
        and excluded from the deploy bridge (mid-train samples used the LIVE
        rows, transient included). Registration fixes all of it: dimensionless
        friction at THEIR lr (kappa_emb = lam/lr_emb), their own winner_step
        schedule group (warmup + cosine, sigma OFF -- the fluctuation never
        earned its keep on embeddings), and inclusion in the deploy bridge."""
        self.emb_trainables = [p["cp"].trainable for p in planes
                               if p.get("cp") is not None and p["cp"].trainable is not None]
        # graph_te (Option C): when the TEs are captured inside the UNet graph, route the embedding
        # shields through the capture-legal path. should_graph_te is False on the bridge path, so the
        # flag stays False and the eager shields run byte-for-byte unchanged. _cm_cap sizes the
        # cm-gate's fixed [dim,cap]/[K,cap] buffers (concord_emb_deflate_modes).
        try:                                    # any failure -> bridge (safe); never crash the run
            # config-FREE: self.config here is the concord config (make_concord_config), NOT the
            # TrainConfig should_graph_te needs -- so use the env/sentinel opt-in signal directly.
            # (True whenever graph_te is opted in; if graph_te doesn't actually engage, the backward
            # runs _apply_group_shield_capture EAGERLY, which is numerically identical to the eager
            # shield -- harmless. Bridge, i.e. sentinel/env absent, => False => original eager path.)
            from modules.util.optimizer.concord_graph import graph_te_opted_in as _gte_on
            _cap_shield = bool(_gte_on())
        except Exception:
            _cap_shield = False
        # read the LOCAL deflate_modes param, not self.emb_deflate_modes -- the latter is still the
        # __init__ default (8) here; it's assigned from deflate_modes ~20 lines below. Must match
        # that assignment so the cm-gate fixed buffer width == the meter's configured mode count.
        _cm_cap = max(0, int(deflate_modes))
        for _tr in self.emb_trainables:
            _tr._use_capture_shield = _cap_shield
            _tr._cm_cap = _cm_cap
        # per-step (timestep, batch-loss) recorder for the timestep-stratified loss analyzer
        # (deconv_loss.py). Opt-in via sentinel CONCORD_LOSS_TS.on; buffered + batch-flushed so
        # there is NO per-step host sync. Meter-only.
        import os as _os_lt
        _lt_root = _os_lt.path.abspath(_os_lt.path.join(_os_lt.path.dirname(__file__), "..", "..", ".."))
        self._loss_ts_pending = []
        self._loss_ts_on = _os_lt.path.exists(_os_lt.path.join(_lt_root, "CONCORD_LOSS_TS.on"))
        self.emb_cores = [t.core for t in self.emb_trainables]
        # control planes aligned with emb_trainables: the plane owns the id->row routing, so
        # the common-mode gate arms THROUGH it (owners by token id, per-plane row order).
        self.emb_cps = [p["cp"] for p in planes
                        if p.get("cp") is not None and p["cp"].trainable is not None]
        self._cm_names_by_plane = [
            [(emb.placeholder if emb is not None else f"base:{_k}") for emb, _k in (p.get("row_map") or [])]
            for p in planes if p.get("cp") is not None and p["cp"].trainable is not None]
        self.emb_row_names = next((
            [(emb.placeholder if emb is not None else f"base:{_k}") for emb, _k in p["row_map"]]
            for p in planes if p.get("row_map")), [])
        self.emb_lr = float(emb_lr)
        self.emb_delay_epochs = max(0.0, float(delay_epochs))
        self.emb_auto_drive = bool(auto_drive)
        self.emb_freq_exponent = max(0.0, float(freq_exponent))
        self.emb_calib_path = calib_path
        self.emb_window_report = bool(window_report)
        self.emb_deflate = bool(deflate)
        self.emb_deflate_gamma = min(0.99, max(0.0, float(deflate_gamma)))
        self.emb_deflate_modes = max(0, int(deflate_modes))   # <= K_TOP; capped again in the meter
        if calib_path:
            import os as _os
            self._cm_sidecar = _os.path.join(_os.path.dirname(calib_path), "concord_commonmode.json")
        for tr in self.emb_trainables:
            tr._track_window = self.emb_window_report   # gate the Sigma||g||^2 accumulator
        if not self.emb_cores:
            return
        lam = (float(self.config.dissipation) if self.config.dissipation is not None
               else float(self.config.gf_consol) * float(self.config.lr))
        kappa_emb = lam / max(self.emb_lr, 1e-12)
        for c in self.emb_cores:
            c.gf_consol = kappa_emb
            # Sighting-clocked dissipation: evidence arrives per SIGHTING for
            # token rows, so evap must tick on evidence steps only -- per-step
            # evap made the effective lambda per unit evidence ~ lambda *
            # sighting_gap (annihilated rare tokens; styles learned, characters
            # did not, 2026-06-12).
            c.grad_activity = True
        print(f"[concord] embedding cores registered: {len(self.emb_cores)} TE plane(s) "
              f"under the controller -- lam={lam:g} @ lr_emb={self.emb_lr:g} -> "
              f"kappa_emb={kappa_emb:.0f} (sighting-clocked); warmup+cosine schedule, "
              f"sigma off; deploy bridge now masks embedding s_fast during sampling")
        if self.emb_deflate:
            self._cm_rearm_from_sidecar()
        # Per-ROW NSR seeding for the embedding token rows (exp52 [epic-williamson]): each
        # row's dissipation seeded from ITS OWN evidence-clocked noise-to-signal ratio --
        # rare-but-consistent tokens get LOW lam (single-sighting evidence survives to
        # consolidate), confused tokens get the ceiling. The measurement is a pure READ of
        # the accumulators the cores already keep (_accum/_power/_seen); the kernel reads a
        # [K] gf buffer selected by SIZE (GF_PER_ROW). Buffer allocated HERE, pre-capture,
        # and only ever copy_()'d after (pointer-stable under any TE-graph mode).
        if noise_seed:
            self._enr_prev = {}
            self._enr_anchor_lam = float(kappa_emb) * float(self.emb_lr)
            for tr in self.emb_trainables:
                c = getattr(tr, "core", None)
                if c is None:
                    continue
                tr._track_window = True          # enables the _power accumulator
                K = int(c.packed_w.shape[0])
                c._gf_consol_buf = torch.full((K,), float(kappa_emb), dtype=torch.float32,
                                              device=c.packed_w.device)
                self._enr_prev[id(tr)] = (tr._accum.detach().float().clone(),
                                          tr._power.detach().float().clone(),
                                          tr._seen.detach().float().clone())
                # validity-scaled chase-admission floor (exp55): arm the per-row
                # kernel variant NOW (pre-capture; flipping later = recapture)
                # with the CURRENT scheduled scalar broadcast -- numerically
                # identical until sighting rates arrive; before_step refreshes
                # values per step (device fill, graph-safe). Leak floor stays
                # the scheduled scalar (its start ~1.0 leaves no scaling room).
                import prototype_packed_b as _pbb
                c._chase_floor_rows = torch.full(
                    (K,), float(_pbb._RATIO_CHASE_FLOOR),
                    dtype=torch.float32, device=c.packed_w.device)
                _pbb.register_row_floors(c.packed_w,
                                         chase_rows=c._chase_floor_rows)
            print(f"[concord] EMB NOISE-SEED armed: per-row kappa on {len(self._enr_prev)} "
                  f"plane(s); anchor lam={self._enr_anchor_lam:g} (median row pins here); "
                  f"commits clamp lam to [epoch-revisit guard, 1.5]; NSR is evidence-clocked (per-sighting). "
                  f"NOTE converged rows read as noisy and drift to the ceiling -- harmless "
                  f"post-consolidation, but their committed lam is not a noise readout.", flush=True)

    @torch.no_grad()
    def _finalize_embedding_calibration(self):
        """Divot calibration: normalize the per-token rate to DISTANCE MOVED
        PER SIGHTING. Over the frozen first epoch the raw gradients accumulate
        coherently (_accum) and each token's batch occurrences are counted
        (_seen); D_i = ||sum||/n_i is the change the data justifies per
        appearance -- the token's own report of how far it still is from its
        place (coherent displacement adds ~n, noise cancels to ~sqrt(n), so a
        converged token reads a small D no matter how often it is seen).

        The drive corrects FREQUENCY ONLY, tempered by the hierarchy exponent:
        drive_i = (median(n)/n_i)^beta, clamped to a decade. A converged-but-
        frequent token moves slowly BECAUSE its D is small (it is NOT boosted
        for having a small total -- the failure mode of normalizing to total
        change); a rare-but-far token gets a frequency boost and its large D
        carries it.

        beta exists because these tokens are HIERARCHICAL (style tokens
        containing object tokens) and frequency is the attribution mechanism:
        in a caption holding both, the shared style residual lands in both
        tokens' gradients, and credit goes to whoever integrates it faster.
        Raw dynamics (beta=0) give the style its rightful n_style/n_obj
        advantage on shared content -- correct attribution, but hot tokens
        fry. Full flattening (beta=1) equalizes per-epoch rates -- attribution
        parity, shared features split by noise (style/object clobbering).
        beta=0.5 equalizes the NOISE motion (D_noise ~ 1/sqrt(n), so
        sqrt(n)*D_noise is constant per token) while justified motion keeps a
        sqrt(frequency) advantage: styles still win shared features, objects
        keep their own content by coherence. The clamp stays tight: a rare
        token's direction estimate is noisy, and amplifying it more than ~5x
        just feeds the friction. Unseen tokens (n=0) keep drive 1.

        PERSISTENCE: the counts are a DATASET property, not a run property --
        measured once, valid until the captions change. A successful
        measurement is saved to the workspace sidecar (emb_calib_path, JSON
        keyed by placeholder, storing n -- the primitive -- so beta stays
        adjustable at load time). A cold start past the divot (resume or
        restart-wrapper segment; the accumulators do NOT round-trip through
        backups) reloads the sidecar instead of re-measuring; uniform drive
        is the fallback only when there is nothing to load."""
        self._emb_drive_applied = True
        beta = max(0.0, float(self.emb_freq_exponent))
        loaded = None
        saved_planes = []
        for plane_idx, tr in enumerate(self.emb_trainables):
            K = tr._seen.numel()
            names = (self.emb_row_names if len(self.emb_row_names) == K
                     else [f"row{i}" for i in range(K)])
            n = tr._seen.float()                         # [K] sightings
            A = tr._accum.float().norm(dim=1)            # [K] justified distance
            P = tr._power.float()                        # [K] incoherent power (window)
            source = "measured this window"
            if not bool((n > 0).any()):
                if loaded is None:
                    loaded = self._load_calibration() or {}
                planes = loaded.get("planes") or []
                tok = (planes[plane_idx].get("tokens", {})
                       if plane_idx < len(planes) else {})
                if not tok:
                    print(f"[concord] embedding calibration TE{plane_idx + 1}: empty "
                          f"accumulator and no saved table -- drive stays uniform")
                    continue
                n = torch.tensor([float(tok.get(nm, {}).get("n", 0.0)) for nm in names])
                A = torch.tensor([float(tok.get(nm, {}).get("A", 0.0)) for nm in names])
                P = torch.tensor([float(tok.get(nm, {}).get("P", 0.0)) for nm in names])
                miss = sorted({nm for nm in names if nm not in tok})
                if miss:
                    print(f"[concord] embedding calibration TE{plane_idx + 1}: "
                          f"{len(miss)} token(s) absent from the saved table keep "
                          f"drive 1 (dataset changed?): {', '.join(miss[:6])}")
                source = (f"restored from sidecar, measured at step "
                          f"{loaded.get('measured_at_step', '?')}")
            live = n > 0
            if not bool(live.any()):
                print(f"[concord] embedding calibration TE{plane_idx + 1}: no live "
                      f"counts -- drive stays uniform")
                continue
            D = A / n.clamp_min(1.0)                     # distance per sighting
            drive = self._emb_drive_from_counts(n, beta)
            tr.set_drive(drive)
            if source == "measured this window":
                saved_planes.append({"te": plane_idx + 1, "tokens": {
                    nm: {"n": round(float(n[i]), 2), "A": float(A[i]),
                         "P": float(P[i])}
                    for i, nm in enumerate(names)}})
            w = drive * n                                # per-epoch rate weight
            order = torch.argsort(torch.where(live, D, torch.full_like(D, -1.0)),
                                  descending=True)       # unseen rows sort last
            live_n = int(live.sum())
            fmt = lambda i: (f"{names[i]}(D={float(D[i]):.3g},n={int(n[i])},"
                             f"d={float(drive[i]):.2f},w={float(w[i]):.0f})")
            far = ", ".join(fmt(int(i)) for i in order[:4])
            near = ", ".join(fmt(int(i)) for i in order[max(0, live_n - 4):live_n].flip(0))
            n_lo = int((drive <= 0.2 + 1e-6).sum())
            n_hi = int((drive >= 5.0 - 1e-6).sum())
            print(f"[concord] embedding calibration TE{plane_idx + 1} [{source}]: "
                  f"drive = (median(n)/n)^{beta:g} over {live_n} live tokens (clamped: "
                  f"{n_lo} at 0.2x, {n_hi} at 5x); w = drive*n = per-epoch rate weight\n"
                  f"          farthest/sighting {far}\n"
                  f"          nearest/sighting  {near}", flush=True)
            if self.emb_window_report:
                self._print_window_report(plane_idx, names, A, P, n)
        if saved_planes and len(saved_planes) == len(self.emb_trainables):
            self._save_calibration(saved_planes)

    @staticmethod
    def _emb_window_stats(C2, P, n):
        """Per-token Wiener posterior around the anchor (init), from the
        coherent-sum power C2 = ||Sigma g||^2, the incoherent power
        P = Sigma ||g||^2, and sighting count n. Model g_i = mu + eps:
            E||Sigma g||^2 = n^2||mu||^2 + n*nu,  E[Sigma||g||^2] = n(||mu||^2 + nu)
        -> ||mu||^2 = (C2 - P)/(n(n-1)),  nu = (P - C2/n)/(n-1).
        Returns (rho, w):
          rho = signal-power fraction ||mu||^2/(||mu||^2+nu) in [0,1] -- the
                Kalman posterior tightness (1 = the data has pinned the
                token's direction; 0 = pure noise, true value could be
                anywhere in the init neighborhood).
          w   = relative window half-width sqrt((1-rho)/(n*rho)) -- the
                fractional uncertainty in the displacement, shrinking as
                1/sqrt(n) (more sightings) and as rho rises (UNet sharpening
                the attractor). Small w = converged."""
        nn = n.clamp_min(2.0)
        mu2 = ((C2 - P) / (nn * (nn - 1.0))).clamp_min(0.0)
        nu = ((P - C2 / nn) / (nn - 1.0)).clamp_min(0.0)
        rho = mu2 / (mu2 + nu + 1e-30)
        w = torch.sqrt((1.0 - rho) / (n.clamp_min(1.0) * rho.clamp_min(1e-6))).clamp_max(99.0)
        seen2 = n >= 2
        rho = torch.where(seen2, rho, torch.zeros_like(rho))
        w = torch.where(seen2, w, torch.full_like(w, 99.0))
        return rho, w

    def _print_window_report(self, plane_idx, names, A, P, n):
        """Surface the per-token posterior: which tokens the divot has
        localized (small w) vs. which are still wandering (wide w)."""
        rho, w = self._emb_window_stats(A * A, P, n)
        live = n >= 2
        if not bool(live.any()):
            print(f"[concord] embedding window TE{plane_idx + 1}: no token seen "
                  f">=2x -- posterior undefined")
            return
        order = torch.argsort(torch.where(live, w, torch.full_like(w, -1.0)),
                              descending=True)            # widest (least sure) first
        live_n = int(live.sum())
        fmt = lambda i: (f"{names[i]}(rho={float(rho[i]):.2f},w={float(w[i]):.2f},"
                         f"n={int(n[i])})")
        wide = ", ".join(fmt(int(i)) for i in order[:4])
        tight = ", ".join(fmt(int(i)) for i in order[max(0, live_n - 4):live_n].flip(0))
        print(f"[concord] embedding window TE{plane_idx + 1}: posterior around init "
              f"(rho=signal fraction, w=relative half-width; small w = data has "
              f"pinned the token)\n"
              f"          least localized {wide}\n"
              f"          most localized  {tight}", flush=True)

    @staticmethod
    def _emb_drive_from_counts(n, beta):
        """drive = (median(n)/n)^beta over live rows, decade clamp, unseen 1."""
        live = n > 0
        med_n = n[live].median()
        return torch.where(
            live, ((med_n / n.clamp_min(1.0)) ** beta).clamp(0.2, 5.0),
            torch.ones_like(n))

    def _save_calibration(self, planes):
        """Write the measured counts to the workspace sidecar. Counts are a
        dataset property; saving makes the calibration once-per-DATASET
        instead of once-per-process (backups don't carry the accumulators,
        and the restart wrapper cycles processes by design)."""
        if not self.emb_calib_path:
            return
        import json
        try:
            with open(self.emb_calib_path, "w", encoding="utf-8") as f:
                json.dump({"version": 1,
                           "beta_at_measure": self.emb_freq_exponent,
                           "window_steps": int(self.emb_delay_steps),
                           "measured_at_step": int(self.step_idx),
                           "planes": planes}, f, indent=1)
            print(f"[concord] embedding calibration saved -> {self.emb_calib_path} "
                  f"(dataset property: resumes/restarts reload it)", flush=True)
        except OSError as e:
            print(f"[concord] embedding calibration save FAILED ({e}); cold "
                  f"resumes will fall back to uniform drive", flush=True)

    def _load_calibration(self):
        if not self.emb_calib_path:
            return None
        import json
        try:
            with open(self.emb_calib_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @torch.no_grad()
    def apply_epoch_window(self, steps_per_epoch):
        """Pin the telescope window to the dataset revisit period (exp-20
        freshness law): alpha_v_fast = 1/(2*steps_per_epoch), so the anchor
        integrates exactly one full pass before motion counts as drift --
        every example votes once. C* is a function of alpha_v, so it is
        re-derived per layer; the telescope-clock consumers (fill ramp,
        probe floor, watchdog arm delay) read config.alpha_v_fast and follow
        automatically. UNet + winner-recipe TEs; frozen-anchor TEs (alpha_v==0) pinned. Call at
        horizon-finalize time, BEFORE the first training step / capture
        (alpha_v_fast and C* are launch-time scalars baked at capture).
        Idempotent across resumes."""
        from prototype_packed_b import compute_drift_cancel_C
        # Stored UNCONDITIONALLY (the divot delay needs steps/epoch even when the
        # telescope window flag is off); the telescope gate is below.
        if steps_per_epoch > 0:
            self.steps_per_epoch = float(steps_per_epoch)
            self.emb_delay_steps = int(round(self.emb_delay_epochs * self.steps_per_epoch))
            if self.emb_cores and self.emb_delay_steps > 0:
                print(f"[concord] embedding release delay: {self.emb_delay_epochs:g} epoch(s) "
                      f"= {self.emb_delay_steps} steps (divot: UNet digs first against the "
                      f"pristine anchors; fresh warmup at release, cosine ends at horizon)",
                      flush=True)
        # v_hat (Adafactor) second-moment EMA window, keyed to the epoch -- INDEPENDENT of the
        # telescope alpha_v window below (applies even with telescope_epoch_window off). Set at
        # train start, BEFORE the first step / capture, like alpha_v. beta2_epoch_window pins
        # beta2 = 1 - 1/steps_per_epoch; else the manual config.beta2. vhat_warmstart is flagged
        # per-layer here and consumed in the backward (init v_row/v_col from the first g^2).
        if steps_per_epoch > 0:
            new_b2 = (1.0 - 1.0 / float(steps_per_epoch)) if self.config.beta2_epoch_window \
                else float(self.config.beta2)
            # bias_correct_v supersedes warm-start (mutually exclusive: warm-start seeds v_hat~g^2 at a
            # non-zero scale, the 1/(1-b2^t) correction assumes a zero start -> running both over-damps).
            want_warm = bool(self.config.vhat_warmstart) and not bool(self.config.bias_correct_v)
            n_warm = 0
            for m in (list(self.layers) + list(self.te_layers)):
                if not getattr(m, "track_adafactor_v", False):
                    continue
                m.adafactor_beta2 = new_b2
                vr = getattr(m, "v_row", None)
                if vr is None:
                    continue
                # Warm-start ONLY a genuinely fresh accumulator (v_row == 0). On the per-epoch
                # exit-42 relaunch v_row is restored non-zero -> skip, so the carried-over v_hat is
                # never clobbered. Host sync is fine here (eager, pre-capture); the blend it arms in
                # the backward is graph-safe and self-zeroing.
                fresh = bool(want_warm and float(vr.abs().sum().item()) == 0.0)
                vr._concord_warm = (torch.ones(1, device=vr.device, dtype=vr.dtype) if fresh else None)
                n_warm += int(fresh)
            print(f"[concord] v_hat beta2 = {new_b2:g} "
                  f"({'1-epoch window' if self.config.beta2_epoch_window else 'manual'}); "
                  f"warm-start = {'on' if want_warm else 'off'}"
                  f"{(' (armed on %d fresh layers)' % n_warm) if want_warm else ''}", flush=True)
        # --- chase (alpha) + leak (alpha_v) timescales, resolved here at horizon-finalize and baked
        # BEFORE capture (alpha / alpha_v_fast / C* are launch-time scalars read from self.* at launch).
        # chase alpha: epoch-window (1/spe) > explicit float > construction default (WINNER 0.1).
        new_alpha = None
        if self.config.chase_epoch_window and steps_per_epoch > 0:
            new_alpha = 1.0 / float(steps_per_epoch)
        elif self.config.chase_alpha and self.config.chase_alpha > 0.0:
            new_alpha = float(self.config.chase_alpha)
        # leak alpha_v: telescope epoch-window (1/2spe) when ON, else the manual Leak Rate knob.
        if self.config.telescope_epoch_window and steps_per_epoch > 0:
            new_av = 1.0 / (2.0 * float(steps_per_epoch))
        else:
            new_av = float(self.config.alpha_v_fast)
        if new_alpha is None and new_av is None:
            return
        old_av = self.config.alpha_v_fast
        if new_av is not None:
            self.config.alpha_v_fast = new_av    # fill ramp / probe floor / watchdog read this
        # UNet layers (all alpha_v>0) always retimed; winner-recipe TEs (alpha_v>0) too; frozen-anchor
        # TEs (alpha_v==0) stay pinned (delicate path -- not touched by the chase/leak retiming).
        for m in (list(self.layers)
                  + [t for t in self.te_layers if getattr(t, "alpha_v_fast", 0.0) > 0.0]):
            if new_alpha is not None:
                m.alpha = new_alpha
            if new_av is not None:
                m.alpha_v_fast = new_av
            m.drift_cancel_C = compute_drift_cancel_C(
                m.alpha, m.alpha_v_fast, mass_preserve=bool(getattr(m, "mass_preserve_v", True)))
        _a = (("%g (1/spe)" % new_alpha) if (new_alpha is not None and self.config.chase_epoch_window)
              else (("%g" % new_alpha) if new_alpha is not None else "default 0.1"))
        print(f"[concord] timescales @ horizon (spe={steps_per_epoch:.0f}): alpha={_a}, "
              f"alpha_v {old_av:g} -> {self.config.alpha_v_fast:g}; C* re-derived per layer "
              f"(fill ramp / probe floor / watchdog follow)", flush=True)

    def read_flow_audit(self):
        """Dissipation flow audit since the last call (one host sync). Returns
        (boil, waste), each None when its denominator is empty:
          boil  = drift-aligned fraction of the killed energy -- gate errors
                  on ESTABLISHED signal (S/A are structurally immune; in-flight
                  kill is the only channel through which learning dissipates);
          waste = killed/(killed+consolidated) energy throughput -- high waste
                  with LOW boil is the lag-tax signature: mass killed before
                  the justification machinery (sig = C*(S-A), which lags the
                  chase) could recognize it. The commit-to-fast-first price."""
        from prototype_packed_b import read_boil
        if not self.layers:
            return None, None
        # SHARED per-device meters: no controller registers per-layer boil buffers for the
        # UNet (the hunting servos were removed), so the kernel's writes land in the shared
        # accumulators and this reader owns their read+zero.
        a, b, c = read_boil(self.layers[0].packed_w.device)
        d = e = f = 0.0
        if not getattr(self, "_boil_shared_seeded", False):
            # Cold-start drain for the SHARED buffer: the first read after process
            # start holds the 2-fast startup self-test (an unregistered toy layer -> shared
            # meters) + the graph-warmup launches, not a training window. Discard it so the
            # first loss line doesn't report the lump as boil/waste.
            self._boil_shared_seeded = True
            self._last_waste = None
            self._last_boil_protected = None
            return None, None
        boil = (a / b) if b > 0 else None
        # slot [2] is NET consolidation flux (chase - eviction refund):
        # clamp at 0 so a net-un-consolidating window reads waste = 1
        # (all throughput killed), never 0/None (which would disengage
        # the waste brake exactly when dissipation dominates).
        c_net = max(c, 0.0)
        waste = (b / (b + c_net)) if (b + c_net) > 0 else None
        self._last_waste = waste                                 # cached for the [loss]-line health log
        self._last_boil_protected = (d / b) if b > 0 else None   # cf-discounted boil (servo gate when protected_boil on)
        self._last_m6a = (e / f) if f > 0 else None   # M6a dissipation-space diversity meter (log-only)
        return boil, waste

    def read_memorization_gap(self):
        """Memorization-gap meter: first-order estimate of (L_deploy - L_live),
        accumulated in the fused backward since the last call (one host sync --
        call at the logging cadence, once per update step). Positive = the live
        weights carry batch-fitted transient (s_fast) the deploy weights don't;
        logged_loss + gap is the deploy-loss estimate that IS comparable across
        friction / gamma-SNR regimes (the live loss is deflated by the
        transient). Trend-accurate while s_fast is small; the exact deploy
        validation at sample time calibrates drift."""
        from prototype_packed_b import read_memgap
        if not self.layers:
            return 0.0
        # SHARED per-device meter (see read_flow_audit): this reader owns the read+zero.
        g = read_memgap(self.layers[0].packed_w.device)
        if not getattr(self, "_gap_shared_seeded", False):
            # Cold-start drain, SHARED buffer (mirror of the per-layer sentinel above): the first
            # read holds the startup self-test + graph-warmup lump, not a training window --
            # returning it seeds the trainer's deploy_smooth EMA on garbage (live symptom:
            # gap=+3e-01 on the first post-resume line, deploy_smooth decaying for ~50 prints).
            self._gap_shared_seeded = True
            return 0.0
        return -g

    @torch.no_grad()
    def set_branch_view(self, which):
        """Hard-negative audit support: swap every 2-fast layer's bf16 forward cache to a
        WEIGHT VIEW -- 'deploy' = s_slow+v_slow (arms dropped), 'L'/'H' = deploy + 2*that arm
        (under the held-out router each arm integrates ~half the data; the factor 2 restores
        full magnitude, so the two views are the two half-data estimates of the weight).
        Views are built by materializing MASKED packed words through the PRODUCTION
        materializer -- no second formula to drift. CALL ONLY at an update boundary: mid-cycle
        the cache holds the frozen cycle-start weight, and restore_branch_views resyncs to
        LIVE, which mid-cycle would break the accumulation freeze. Unsupported under fused
        matmul (no per-layer cache; rematerialized from packed every forward). Returns the
        number of layers swapped -- 0 means nothing auditable (fused mode or not 2-fast)."""
        import prototype_packed_b as _pb
        if _pb._FUSED_MATMUL:
            return 0
        from prototype_packed_2fast import materialize_packed2_bf16
        n = 0
        for m in self.layers:
            wb = getattr(m, "_bf16_weight_buf", None)
            if wb is None or not hasattr(m, "arm_row_exp") or wb.shape != m.packed_w.shape:
                continue
            # deploy component: arms masked off (low 16 bits = s_slow | v_slow)
            materialize_packed2_bf16(m.packed_w & 0x0000FFFF, m.row_exp, m.col_exp,
                                     m.arm_row_exp, m.arm_col_exp, out=wb,
                                     mantissa_bias=m.MANTISSA_BIAS)
            if which != "deploy":
                # keep only that arm's byte (e_L = bits 24-31 -> mask is the sign-safe
                # int32 form of 0xFF000000; e_H = bits 16-23)
                arm_mask = -16777216 if which == "L" else 0x00FF0000
                scratch = torch.empty_like(wb)
                materialize_packed2_bf16(m.packed_w & arm_mask, m.row_exp, m.col_exp,
                                         m.arm_row_exp, m.arm_col_exp, out=scratch,
                                         mantissa_bias=m.MANTISSA_BIAS)
                wb.add_(scratch, alpha=2.0)
                del scratch
            n += 1
        return n

    @torch.no_grad()
    def restore_branch_views(self):
        """Resync every layer's forward cache to the LIVE weight (identical to what the update
        step just wrote -- update-boundary-only, see set_branch_view). Pairs with it."""
        for m in self.layers:
            if hasattr(m, "_resync_weight_buf"):
                m._resync_weight_buf()

    def branch_view_diag(self):
        """One-line WHY for a set_branch_view that swapped zero layers: which precondition
        failed -- fused matmul (no per-layer cache; views can't be packed-encoded, 2*e_L
        overflows int8), no arms (not two_fast), or missing/mismatched forward caches."""
        import prototype_packed_b as _pb
        n_arm = sum(1 for m in self.layers if hasattr(m, "arm_row_exp"))
        n_buf = sum(1 for m in self.layers
                    if hasattr(m, "arm_row_exp")
                    and getattr(m, "_bf16_weight_buf", None) is not None
                    and m._bf16_weight_buf.shape == m.packed_w.shape)
        return (f"fused_matmul={bool(_pb._FUSED_MATMUL)} layers={len(self.layers)} "
                f"with_arms={n_arm} with_cache={n_buf}")

    @torch.no_grad()
    def _csnr_step(self, timesteps, alphas_cumprod):
        """Graph-native CSNR driving (no-op unless the meter is enabled). Arms ~once per epoch,
        pools the accumulation micro-batches' timesteps, and flushes ONE observation per optimizer
        step (the buffers' pooled batch-mean + its timesteps) into the collector. GPU-validation
        points when enabled: the arm cadence and the flush-at-accumulation-boundary buffer timing."""
        if not getattr(self, "_csnr_bufs", None):
            return
        if self._csnr is None:                              # lazy build once alphas_cumprod is known
            # package spelling: csnr_meter lives in modules/util/optimizer/ (NOT the concord/
            # subdir that the sys.path inserts cover) -- the bare name is unresolvable here
            from modules.util.optimizer.csnr_meter import CSNRCollector
            K = sum(int(c.numel()) for _, c in self._csnr_bufs)
            self._csnr = CSNRCollector([], int(alphas_cumprod.numel()), alphas_cumprod=alphas_cumprod,
                                       window=self._csnr_window, K=K, source="kernel")
        spe = int(round(self.steps_per_epoch)) or self._csnr_window
        if (not self._csnr.active()) and (self.step_idx - self._csnr_last_arm) >= spe:
            for _b, _ in self._csnr_bufs:                   # kernel writes every step -> start clean
                _b.zero_()
            self._csnr.arm(); self._csnr_last_arm = self.step_idx; self._csnr_pool = []
        if self._csnr.active():
            if len(self._csnr_pool) >= self._csnr_accum:    # a full optimizer step completed -> flush
                self._csnr_flush()
            self._csnr_pool.append(timesteps.detach())

    @torch.no_grad()
    def _csnr_flush(self):
        """Assemble the feat-only batch-mean per selected layer (sum/count), pool timesteps, feed
        one observation, and on a KEPT window store the curve. The sampler connection and the
        loss-prediction readout (vs _kloss) are deliberate follow-ups in the trainer."""
        us = []
        for buf, coord in self._csnr_bufs:
            k = int(coord.numel()); cnt = float(buf[k].item())
            us.append((buf[:k] / cnt).float().cpu() if cnt > 0 else torch.zeros(k))
            buf.zero_()                                     # reset for the next optimizer step's pool
        if self._csnr_pool:
            ts = torch.cat(self._csnr_pool); g0 = self._csnr.gen
            self._csnr.observe(ts, sketch=torch.cat(us))
            if self._csnr.gen > g0:                         # a window was kept -> a fresh curve
                self._csnr_curve = self._csnr.emit()        # (curve, gen)
                print(f"[concord] CSNR curve updated (gen {self._csnr_curve[1]}); "
                      f"loss-prediction check pending (compare to _kloss in the trainer)", flush=True)
                # Dump the measured curve (compact, 8-bin) so the armgap x SNR harness can join
                # per-pair timesteps to LIVE gradient-SNR, not just schedule SNR. Meter-only.
                try:
                    import os as _os4, json as _json4
                    _cvv = self._csnr_curve[0].float().cpu()
                    _T = int(_cvv.numel()); _nb = 8
                    _edg = [int(round(i * _T / _nb)) for i in range(_nb + 1)]
                    _bins = [float(_cvv[_edg[i]:max(_edg[i] + 1, _edg[i + 1])].mean()) for i in range(_nb)]
                    _pc = _os4.path.join(self.config.workspace_dir, "concord_csnr_curve.jsonl")
                    with open(_pc, "a", encoding="utf-8") as _fc:
                        _fc.write(_json4.dumps({"step": int(self.step_idx),
                                                "gen": int(self._csnr_curve[1]),
                                                "n_bins": _nb, "edges": _edg, "snr_bins": _bins}) + "\n")
                except Exception:
                    pass
        self._csnr_pool = []

    def on_timesteps(self, timesteps, alphas_cumprod):
        """gamma-SNR dissipation modulation (opt-in via optimizer.autotune_gamma_snr).

        Also stashes the batch's timesteps (self._last_timesteps) for the Kalman loss
        meter in GenericTrainer -- the t-conditional baseline needs to know WHICH
        timesteps produced each loss observation. Stash-only; no behavior change.

        min-SNR-gamma's loss weight w(t) = min(snr, gamma)/snr is a hand-designed
        PRIOR for "limit the influence of the conflicting high-SNR gradient
        stream". The gate's coherence meter measures that conflict directly and
        the autotuner turns it into a base friction; this hook adds the
        timestep-resolved shape on top:

            m       = mean_batch( max(1, snr_i / knee) )   (inverse of w, knee at gamma)
            kappa_t = min(kappa_base * m, _LAM_MOD_CAP / lr)

        Batches at timesteps min-SNR would down-weight get proportionally MORE
        dissipation instead -- the loss stays unweighted, the regularizer absorbs
        the role, and the overall strength is the autotuned base rather than a
        hand-tuned gamma.

        Exogenous by construction: the modulation input is the sampler's timestep
        draw, not the meter, so this does NOT reintroduce the exp-11 closed loop.
        While the tuner is probing (committed is None) the hook is silent and the
        probe runs at the clean constant probe kappa -- calibration conditions
        match the table.

        Graph-native: writes each UNet layer's gf_consol device buffer directly
        (device-to-device 0-dim copy, no host sync); the captured backward reads
        the buffers at replay. The host mirror (_gf_consol_value) keeps the
        UNMODULATED base. Call after the batch's timesteps are sampled and before
        the backward / graph replay. Frozen-anchor TEs (alpha_v==0) are never
        modulated; winner-recipe TEs (alpha_v>0) are, same as the UNet.
        """
        self._last_timesteps = timesteps.detach()      # kalman loss meter reads this (stash-only)
        # One-time dump of the noise schedule (alphas_cumprod) so the offline armgap x SNR harness
        # can map timestep -> schedule SNR = abar/(1-abar) without touching the dataset. Meter-only;
        # captures the ACTUAL schedule (incl. any zero-terminal-SNR rescale), not a reconstructed one.
        if not getattr(self, "_schedule_dumped", False):
            self._schedule_dumped = True
            try:
                import os as _os3, json as _json3
                _p = _os3.path.join(self.config.workspace_dir, "concord_schedule.json")
                with open(_p, "w", encoding="utf-8") as _fsch:
                    _json3.dump({"alphas_cumprod": alphas_cumprod.detach().float().cpu().tolist()}, _fsch)
            except Exception:
                pass
        self._csnr_step(timesteps, alphas_cumprod)     # graph-native CSNR meter (no-op unless enabled)
        knee = self.config.autotune_gamma_snr
        if (not getattr(self.config, "autotune_gamma_snr_on", True)
                or knee is None or float(knee) <= 0.0):
            # Selector OFF, or no/degenerate knee: skip the modulation. before_step has already
            # written gf_consol_buf = base(/servo) * fill_ramp, so the un-modulated lam stands --
            # NOT scaled by SNR, NOT clamped to lam=1. (knee=0 used to mean snr/0 -> inf -> every
            # step pinned at the lam=1 cap; <= 0 now reads as OFF.)
            if not self._snr_mod_announced:
                self._snr_mod_announced = True
                print("[concord] gamma-SNR dissipation modulation OFF "
                      "(base/servo lam applies directly; not capped to lam=1)")
            return
        if self.autotuner is not None:
            base = self.autotuner.committed
            if base is None:
                return                      # probe / re-probe window: stay clean
        else:
            base = self.config.gf_consol    # fixed-friction run: modulate the config base
        if base is None or float(base) <= 0:
            return
        ac = alphas_cumprod.to(timesteps.device)[timesteps.long()].float()
        snr = ac / (1.0 - ac).clamp_min(1e-8)
        mod = (snr / float(knee)).clamp_min(1.0).mean()
        cap = self._LAM_MOD_CAP / max(self.config.lr, 1e-12)   # UNet cap, for the announcement below
        if not self._snr_mod_announced:
            self._snr_mod_announced = True
            print(f"[concord] gamma-SNR dissipation modulation ON: knee={float(knee):g}, "
                  f"base kappa={float(base):.0f}, cap lam={self._LAM_MOD_CAP:g} "
                  f"(kappa <= {cap:.0f} @ lr={self.config.lr:g})")
            if float(base) > cap:
                print(f"[concord] WARNING: base kappa {float(base):.0f} EXCEEDS the "
                      f"gamma-SNR cap ({cap:.0f}, lam={self._LAM_MOD_CAP:g}) -- since the "
                      f"modulation only scales UP and then clamps, every modulated step "
                      f"runs at the cap: effective lam = {self._LAM_MOD_CAP:g} < your "
                      f"base. Lower the base into the plateau (exp 21: lam* ~ 0.5-1.0) "
                      f"or disable gamma-SNR to run above it.")
        servo = getattr(self.autotuner, "per_layer", False)
        # Winner-recipe TEs are modulated like the UNet; frozen-anchor TEs (alpha_v==0) are not.
        _mod_layers = self.layers + [m for m in self.te_layers
                                     if getattr(m, "alpha_v_fast", 0.0) > 0.0]
        # PER-LAYER stability cap: the kernel applies evap_frac = lr*gf_consol*(1-coh) with each
        # layer's OWN lr, so the lam <= _LAM_MOD_CAP cap must use THAT lr -- config.lr for the UNet,
        # te_lr/te2_lr for each winner TE. A single config.lr cap would over-clamp a low-LR TE and,
        # worse, under-clamp a high-LR TE (lam_te could exceed _LAM_MOD_CAP). Compose the global
        # timestep shape (mod) with each layer's base (servo: per-layer _gf_consol_value mirror; else
        # the global base) and the fill ramp, then clamp to the per-layer cap.
        _lr_of = {id(m): self.config.lr for m in self.layers}
        for _te_lyr, _te_lr in self.te_groups:
            for _m in _te_lyr:
                _lr_of[id(_m)] = _te_lr
        for layer in _mod_layers:
            _cap_l = self._LAM_MOD_CAP / max(_lr_of.get(id(layer), self.config.lr), 1e-12)
            _rows = getattr(layer, "_gf_consol_rows", None)
            if _rows is not None and _rows.numel() == layer._gf_consol_buf.numel():
                # row-aware (audit C-4): modulate the seeder's row kappas, not a flat base
                layer._gf_consol_buf.copy_(
                    (mod * _rows * self._current_fill_ramp).clamp_max(_cap_l))
                continue
            # the layer's committed kappa (seeder or config both land in the host
            # mirror via the property) -- 'base' is the raw config value and would
            # silently discard seeder commits now that the hunting servos are gone
            _base_l = float(getattr(layer, "_gf_consol_value", base))
            layer._gf_consol_buf.copy_(
                (mod * _base_l * self._current_fill_ramp).clamp_max(_cap_l))

    def _emb_commonmode_meter(self):
        """Log-only, once per epoch: SVD the epoch INCREMENT of each emb core's coherent
        accumulator (_accum is never zeroed -> diff against our own copy). Reports (a) how much
        of the accumulated update's energy lives in the top-k shared directions, (b) the per-row
        fraction of each token's step inside that subspace (subject vs tag split when the quality
        shield is wired), (c) the subspace's stability vs the previous epoch. This is the
        dry-run for the deflation shield: big + stable common mode on subject rows = the knob
        would earn its keep. When concord_emb_deflate is ON it is also the ARMING pass:
        stable+owned components -> per-row shrink gate on the trainable, persisted to the
        concord_commonmode.json sidecar (owners by token NAME; cores reset every exit-42
        segment). Eager context (epoch boundary); log-only unless deflate is enabled.
        NOTE: iterates emb_trainables (ConcordPackedEmbedding, which owns _accum/_seen/
        _quality_*) -- emb_cores holds the INNER ConcordLinearPackedB, which has none of them."""
        import os as _os
        import torch as _t
        if _os.environ.get("CONCORD_COMMONMODE_METER", "1") == "0":
            return
        # 8 components: the measured accumulator carries ~10+ owned axes (quality tags + 2-3
        # fragments per named concept); top-4 was leaving st/usc/ld/stl + the yaSattra/akama
        # fragments unexamined below the cut. Cost is nil (SVD already computes the full
        # spectrum; the per-step gate grows to [dim,8] projections ~ 80 MFLOPs).
        K_TOP = 8
        # How many owned modes to actually ARM (shrink) this epoch. The meter still examines
        # and prints all K_TOP; only the top _n_modes ELIGIBLE (energetic+stable+owned, in
        # energy order) are armed. Knob: concord_emb_deflate_modes.
        _n_modes = max(0, min(K_TOP, int(getattr(self, "emb_deflate_modes", K_TOP))))
        _side = {"step": int(self.step_idx), "gamma": float(self.emb_deflate_gamma), "cores": {}}
        for _i, core in enumerate(getattr(self, "emb_trainables", None) or []):
            A = getattr(core, "_accum", None)
            if A is None:
                continue
            with _t.no_grad():
                prev = self._cm_prev.get(id(core))
                A_ep = (A - prev[0]) if prev is not None else A.clone()
                tot = float(A_ep.norm())
                if tot < 1e-12:
                    print(f"[concord-commonmode:emb{_i}] t={self.step_idx}: no accumulation this "
                          f"window (divot or starved) -> skipped", flush=True)
                    continue
                try:
                    _U, S, Vh = _t.linalg.svd(A_ep.float(), full_matrices=False)
                except Exception as _e:
                    print(f"[concord-commonmode:emb{_i}] svd failed ({type(_e).__name__})", flush=True)
                    continue
                V = Vh[:K_TOP].T                                  # [dim, k]
                e = S.square()
                esh = (e[:K_TOP] / e.sum().clamp_min(1e-30))
                rn = A_ep.norm(dim=1)
                live = rn > 1e-12
                frac = (A_ep @ V).norm(dim=1) / rn.clamp_min(1e-12)
                f_all = frac[live]
                # subject/tag split when the quality shield is wired (mask: 1=subject, 0=tag)
                split = ""
                qm = getattr(core, "_quality_subject_mask", None)
                if getattr(core, "_quality_Q", None) is not None and qm is not None:
                    subj = live & (qm.reshape(-1) > 0.5); tag = live & (qm.reshape(-1) <= 0.5)
                    if bool(subj.any()) and bool(tag.any()):
                        split = (f" | subj p50={frac[subj].median():.2f} "
                                 f"tag p50={frac[tag].median():.2f}")
                # epoch-over-epoch subspace stability: mean cos of principal angles
                stab = ""
                if prev is not None and prev[1] is not None and prev[1].shape == V.shape:
                    stab = (f" | vs prev: top1 |cos|={float((V[:, 0] @ prev[1][:, 0]).abs()):.2f}, "
                            f"basis overlap={float(_t.linalg.svdvals(V.T @ prev[1]).mean()):.2f}")
                print(f"[concord-commonmode:emb{_i}] t={self.step_idx} live_rows={int(live.sum())} "
                      f"top{K_TOP} energy={[round(float(x), 3) for x in esh]} (sum={float(esh.sum()):.2f})"
                      f" | row-frac in top{K_TOP}: p50={f_all.median():.2f} p90={f_all.quantile(0.9):.2f}"
                      f"{split}{stab}", flush=True)
                # ATTRIBUTION: each component's owners. u_i (per-row loading) is free in the SVD;
                # affinity = fraction of the row's OWN update along v_i ("who is ABOUT this") vs
                # magnitude = A_t . v_i ("who DRIVES it", exposure-dominated). The future deflation
                # gate keys on affinity (n-shrunk); this names the components so it can be sanity-
                # checked against the probe-measured axes (style -> masterpiece/uscld/...).
                _byp = getattr(self, "_cm_names_by_plane", None) or []
                names = (_byp[_i] if _i < len(_byp) and _byp[_i]
                         else (getattr(self, "emb_row_names", None) or []))
                _cps = getattr(self, "emb_cps", None) or []
                _tids = (getattr(_cps[_i], "train_tids", []) or []) if _i < len(_cps) else []
                seen = getattr(core, "_seen", None)
                def _nm(j):
                    s = names[j] if j < len(names) else f"row{j}"
                    n = int(seen[j]) if seen is not None else -1
                    return s, n
                loud = rn > rn[live].median() * 0.25 if bool(live.any()) else live   # noise-row floor
                comps = []
                for _c in range(min(K_TOP, V.shape[1])):
                    ld = A_ep @ V[:, _c]
                    aff = (ld / rn.clamp_min(1e-12)).abs() * (live & loud)
                    own = _t.argsort(-aff)[:4].tolist()
                    drv = _t.argsort(-ld.abs())[:3].tolist()
                    o = ", ".join(f"{_nm(j)[0]}(a={float(aff[j]):.2f},n={_nm(j)[1]})" for j in own)
                    d = ", ".join(_nm(j)[0] for j in drv)
                    print(f"[concord-commonmode:emb{_i}]   c{_c} e={float(esh[_c]):.3f} "
                          f"owners: {o} | drivers: {d}", flush=True)
                    # ---- arming eligibility (only used when concord_emb_deflate is ON) ----
                    if not self.emb_deflate or float(esh[_c]) < 0.05:
                        continue
                    # stability: does this DIRECTION exist in last epoch's top-k? Components
                    # permute by energy rank between windows (measured: four near-equal single-
                    # token axes), so match against ALL prev columns, not index-to-index.
                    ref = (prev[1] if prev is not None and prev[1] is not None
                           else self._cm_side_prev.get(_i))
                    if ref is None:
                        continue                                   # no reference yet -> hands off
                    if float((ref.to(V.device).T @ V[:, _c]).abs().max()) < 0.75:
                        continue                                   # young/unstable -> hands off
                    owners = _t.nonzero((aff >= 0.6)).reshape(-1).tolist()
                    if not (1 <= len(owners) <= max(8, int(0.02 * A_ep.shape[0]))):
                        continue                                   # unowned or flat -> hands off
                    if len(comps) >= _n_modes:
                        continue                                   # mode-count cap (concord_emb_deflate_modes); rest still metered
                    comps.append({"v": [round(float(x), 6) for x in V[:, _c].tolist()],
                                  "energy": round(float(esh[_c]), 4),
                                  "owners": [_nm(j)[0] for j in owners],
                                  "owners_tid": [int(_tids[j]) for j in owners if j < len(_tids)]})
                if self.emb_deflate:
                    if self._cm_arm(_i, comps):
                        desc = "; ".join(",".join(c["owners"][:3]) + f" e={c['energy']:.2f}" for c in comps)
                        print(f"[concord-commonmode-GATE:emb{_i}] ARMED {len(comps)} component(s): {desc} "
                              f"(gamma={self.emb_deflate_gamma:g}, cap={_n_modes})", flush=True)
                    else:
                        print(f"[concord-commonmode-GATE:emb{_i}] no eligible components "
                              f"(unstable/unowned/flat/small) -> gate idle", flush=True)
                    self._cm_state[_i] = comps
                # sidecar: V always (cross-segment stability for the meter), comps when armed
                _side["cores"][str(_i)] = {"V": [[round(float(x), 6) for x in V[:, _c].tolist()]
                                                 for _c in range(V.shape[1])],
                                           "comps": comps}
                self._cm_prev[id(core)] = (A.detach().clone(), V.detach().clone())
        if self._cm_sidecar and _side["cores"]:
            try:
                import json as _json
                with open(self._cm_sidecar, "w", encoding="utf-8") as f:
                    _json.dump(_side, f)
            except OSError as _e:
                print(f"[concord-commonmode] sidecar write failed ({_e})", flush=True)

    @torch.no_grad()
    def _emb_nsr_seed(self):
        """Per-row NSR commit for the embedding token rows (exp52 + exp55). Windowed
        DELTAS of the cores' own accumulators (snapshot discipline like the commonmode
        meter -- these accumulators have other consumers and are never zeroed here): per
        row, NSR = (dP/dn) / ||dA/dn||^2 - 1 over the row's OWN sightings this window.
        LAW (exp55): whitened per-sighting base re-anchored
        each seeding (median measured row's PER-SIGHTING lam = the anchor), converted to
        the update clock by the row's own sighting rate -- equal decay per SIGHTING, so a
        rare row's innovations survive the updates it spends waiting (the update-clock
        commit cost rare tokens +60pp of real learning on the CPU harness, and was worse
        than flat everywhere). Rows with dn < 8 keep the anchor
        (estimator saturates at ~n-1). Commit lam clamp [lam_lo, 1.5]: ceiling per exp52
        (1.7+ diverges under the bracket arm; converged rows drift to the ceiling by
        construction); floor = the epoch-revisit guard, same law as NoiseScaleSeeder's
        (window 1/lam <= epoch/4 in UPDATES -- the decay clock is updates even though
        deposits are per-sighting; exp53 cliff: past the epoch, a recurring example
        straddles both arms within one window and self-corroborates)."""
        lam0 = self._enr_anchor_lam
        anchor_kap = lam0 / max(self.emb_lr, 1e-12)
        _spe = int(round(self.steps_per_epoch)) if self.steps_per_epoch > 0 else 0
        lam_lo = min(4.0 / _spe, 1.0) if _spe > 0 else 0.02
        for tr in (getattr(self, "emb_trainables", None) or []):
            prev = self._enr_prev.get(id(tr))
            c = getattr(tr, "core", None)
            if prev is None or c is None:
                continue
            A = tr._accum.detach().float()
            P = tr._power.detach().float()
            S = tr._seen.detach().float()
            dA, dP, dS = A - prev[0], P - prev[1], S - prev[2]
            self._enr_prev[id(tr)] = (A.clone(), P.clone(), S.clone())
            meas = dS >= 8.0
            if not bool(meas.any()):
                continue
            n = dS.clamp_min(1.0)
            eg2 = (dA / n[:, None]).pow(2).sum(dim=1)     # ||mean g||^2 per row
            g2 = dP / n                                    # E||g||^2 per row, per sighting
            nsr = (g2 / eg2.clamp_min(1e-30) - 1.0).clamp_min(1e-3)
            vals = nsr[meas]
            med = float(vals.median())
            buf = c._gf_consol_buf
            # WHITENED per-sighting commit, re-anchored each seeding (median measured
            # row's per-sighting lam = the anchor). The PRODUCTION emb evaporation is
            # SIGHTING-CLOCKED in the kernel (grad-activity gated -- the registration
            # line prints 'sighting-clocked'), so a per-sighting commit already IS the
            # evidence clock here: exp55's update-clock conversion (lam * rate) applies
            # only to update-clocked kernels like its harness, and applying it on a
            # sighting-clocked kernel DOUBLE-discounts rare rows. No update-clock
            # epoch-guard either (the revisit defense concerns SHARED weights; an emb
            # row learning its few examples is the job -- exp55 memnoise: shared-weight
            # mem stays low with the validity floors carrying admission).
            Cw = lam0 * max(med, 1e-6)
            lam_sight = (Cw / nsr.clamp_min(1e-3)).clamp(0.02, 1.5)   # sighting-clock clamps (exp55)
            rate = (dS / max(float(_spe), 1.0)).clamp_min(1e-9)       # sightings per update
            self._enr_rate[id(tr)] = rate                              # validity FLOORS refresh from this
            kap = lam_sight / max(self.emb_lr, 1e-12)
            if buf.numel() != kap.numel():                # scalar buf: seeding not armed for this core
                continue
            buf.copy_(torch.where(meas, kap,
                                  torch.full_like(kap, anchor_kap)).to(buf.dtype))
            lm = lam_sight[meas]
            rt = rate[meas]
            print(f"[concord-emb-nsr] t={self.step_idx} (per-sighting, whitened): rows "
                  f"{int(meas.sum())}/{int(meas.numel())} | "
                  f"NSR min/med/max={float(vals.min()):.1f}/{med:.1f}/{float(vals.max()):.1f} | "
                  f"rate p10/50/90={float(rt.quantile(0.1)):.4f}/{float(rt.median()):.4f}/"
                  f"{float(rt.quantile(0.9)):.4f} | "
                  f"lam_sight min/med/max={float(lm.min()):.3f}/{float(lm.median()):.3f}/{float(lm.max()):.3f}",
                  flush=True)

    def _cm_arm(self, plane_idx, comps):
        """Delegate arming to the CONTROL PLANE (it owns the id->row routing; owners travel
        as token ids, which are stable across segments and plane row orders). Returns True
        if the plane armed at least one owner row."""
        cps = getattr(self, "emb_cps", None) or []
        if plane_idx >= len(cps):
            return False
        return cps[plane_idx].arm_common_mode_gate(comps, self.emb_deflate_gamma) > 0

    def _cm_rearm_from_sidecar(self):
        """Segment start: cores were rebuilt from the materialized table (all gate state lost).
        Reload the sidecar -> restore prev-V for the meter's stability test AND re-arm the
        gates immediately (owners mapped by NAME through the current emb_row_names)."""
        import json as _json, os as _os
        import torch as _t
        self._cm_side_prev = {}
        self._cm_state = {}
        p = self._cm_sidecar
        if not p or not _os.path.exists(p):
            print("[concord-commonmode-GATE] no sidecar yet -> gate arms at the first epoch boundary",
                  flush=True)
            return
        try:
            with open(p, encoding="utf-8") as f:
                d = _json.load(f)
        except (OSError, ValueError) as _e:
            print(f"[concord-commonmode-GATE] sidecar unreadable ({_e}) -> starting cold", flush=True)
            return
        for _i, tr in enumerate(getattr(self, "emb_trainables", None) or []):
            cd = (d.get("cores") or {}).get(str(_i))
            if not cd:
                continue
            if cd.get("V"):
                self._cm_side_prev[_i] = _t.tensor(cd["V"], dtype=_t.float32).T   # [dim, k]
            comps = cd.get("comps") or []
            for c in comps:                    # back-compat: name-only sidecars -> parse 'base:<id>'
                if not c.get("owners_tid"):
                    c["owners_tid"] = [int(nm.split(":", 1)[1]) for nm in (c.get("owners") or [])
                                       if isinstance(nm, str) and nm.startswith("base:")
                                       and nm.split(":", 1)[1].isdigit()]
            if comps and self._cm_arm(_i, comps):
                self._cm_state[_i] = comps
                print(f"[concord-commonmode-GATE:emb{_i}] re-armed from sidecar "
                      f"(step {d.get('step')}): {len(comps)} component(s), "
                      f"owners e.g. {','.join(comps[0]['owners'][:3])}", flush=True)
            else:
                # V-only sidecar (meter ran while deflate was off, or nothing qualified last
                # boundary): stability reference loaded, nothing armed -- say so, loudly enough
                # that "gate on but silent" is distinguishable from "gate not wired".
                print(f"[concord-commonmode-GATE:emb{_i}] sidecar loaded (V@step {d.get('step')}, "
                      f"no armed components) -> gate evaluates at the next epoch boundary", flush=True)

    def before_step(self):
        """BEFORE forward/backward: advance the winner schedule onto the layer device
        tensors (lr / sigma / coherence floors) that the fused backward reads."""
        from concord_winner import winner_step
        if self._autotune_pending:
            self._build_autotuner()
        # GRADIENT-ACCUMULATION FAST-PATH: before_step is invoked `accum` times
        # per update with the SAME step_idx (after_step advances it once per
        # update). Everything below is keyed to step_idx and re-launches ~800
        # per-layer EAGER kernels (winner_step schedule writes, fill-ramp,
        # per-layer LAMB maintenance, quality-basis SVDs) with identical inputs
        # -- the dominant accumulation overhead (measured ~75ms x (accum-1) of
        # redundant launches per update, >1s/update at accum=16 on WDDM). Run it
        # ONCE per update: the device buffers persist across the cycle's replays
        # and the captured backward reads them every micro-step, so the deployed
        # weight is unchanged. One intended semantic shift: LAMB's norm window
        # becomes per-UPDATE (the correct trust-ratio granularity) rather than
        # per-micro-step (which zeroed the accumulators mid-cycle).
        if self.step_idx == getattr(self, '_last_sched_idx', -1):
            return
        self._last_sched_idx = self.step_idx
        # PREQ RE-REGISTRATION (self-heal). The kernel's WRITE_PREQ + preq_buf are BAKED at CUDA-graph
        # CAPTURE from _lookup_preq(packed_w). Sampling/backup drop + recapture the graph, and any event
        # that moved a layer's packed_w data_ptr since it was last registered leaves _lookup_preq None ->
        # WRITE_PREQ False baked -> that layer's gap is never metered (the "stale-reg" dropout; ~651/722
        # observed). gf_consol survives capture because its buffer is a permanent member; preq comes from
        # the data_ptr-keyed registry, so re-register any stale layer HERE (eager, before the captured
        # step) and the NEXT recapture bakes it in. Cheap: one dict lookup per layer, re-register only the
        # (rare) stale ones; prints once per reallocation event, then quiet.
        if getattr(self, "_preq_heal", True):
            try:
                # PURGE-AND-REBUILD by module identity, never is-None healing: after a
                # reallocation a layer can inherit another layer's stale data_ptr, so a
                # non-None lookup proves nothing (observed live: 'preq=680 nsr=0' -- the
                # nsr heal was blinded by collisions while preq re-registered). Stale
                # keys are also a device-assert bomb when sizes differ. Rebuild cost is
                # ~2k dict inserts per step -- noise.
                import prototype_packed_2fast as _p2f
                import prototype_packed_b as _ppb
                _p2f.reregister_all_preq(
                    [(_m.packed_w, _m._preq_meter) for _m in (self.layers + self.te_layers)
                     if getattr(_m, "_preq_meter", None) is not None])
                _p2f.reregister_all_nsr(
                    [(_m.packed_w, _m._nsr_buf, _m._nsr_idx) for _m in (self.layers + self.te_layers)
                     if getattr(_m, "_nsr_buf", None) is not None])
                _ppb.reregister_all_row_floors(
                    [(c.packed_w, c._chase_floor_rows, None)
                     for _tr in (getattr(self, "emb_trainables", None) or [])
                     for c in [getattr(_tr, "core", None)]
                     if c is not None and getattr(c, "_chase_floor_rows", None) is not None])
            except Exception:
                self._preq_heal = False
        # Noise-scale seeder (set-don't-hunt, exp50): lazy-built on the FIRST before_step --
        # which precedes the first CUDA-graph capture, so the sketch buffers register
        # pre-capture (later reallocations are covered by the self-heal above). The seeder
        # is the ONLY dissipation controller; it owns per-layer kappa.
        # NSR seeder build (UNet + optionally the winner TEs). The TE seeder build is INDEPENDENT of
        # self.layers -- a frozen-UNet run (unet.train=False -> self.layers==[]) with the TE servo
        # opted in must still arm the TE seeders (review finding: otherwise the "TE-ROW-SERVO ON" log
        # overclaims while nothing arms). Shared locals (_spe/_win/_nsr_sc/_pr) don't depend on the
        # UNet, so compute them once and gate each seeder separately.
        _need_unet_seeder = (getattr(self, "_nsr_seeder", None) is None and bool(self.layers))
        _need_te_seeder = (getattr(self, "_nsr_seeder_te", None) is None
                           and getattr(self, "_te_row_servo", False))
        if (_need_unet_seeder or _need_te_seeder) \
                and bool(getattr(self.config, "noise_seed_servo", False)):
            _spe = int(round(self.steps_per_epoch)) if self.steps_per_epoch > 0 else 0
            _per = max(1, int(getattr(self.config, "autotune_servo_per_epoch", 3) or 3))
            _win = max(16, _spe // _per) if _spe > 0 else 128
            _nsr_sc = None
            import os as _os
            if self.workspace_dir:
                _nsr_sc = _os.path.join(self.workspace_dir, "concord_nsr.json")
            elif self.emb_calib_path:
                _nsr_sc = _os.path.join(_os.path.dirname(self.emb_calib_path), "concord_nsr.json")
            _pr = bool(getattr(self.config, "nsr_per_row", True))
            if _pr and not bool(getattr(self.config, "heldout_router", False)):
                # the arm meter's R is a CROSS-SPLIT meter only under the router; unrouted, the gap
                # is the rate-bracket differential and the per-row law would seed from a meaningless
                # quantity
                _pr = False
                print("[concord] NSR PER-ROW forced OFF: heldout_router is off, so the arm gap is "
                      "not a data-split meter (enable the router to use the per-row law)", flush=True)
            if _need_unet_seeder:
                if _nsr_sc is None:
                    print("[concord-nsr] no workspace dir known -> sidecar persistence OFF "
                          "(seeder still runs; lams re-derive each segment)", flush=True)
                _lam0 = float(self.config.gf_consol) * float(self.config.lr)
                self._nsr_seeder = NoiseScaleSeeder(self.layers, float(self.config.lr), _lam0,
                                                    _win, sidecar=_nsr_sc, epoch_updates=_spe,
                                                    per_row=_pr, now_t=self.step_idx)
            if _need_te_seeder:
                # SEPARATE seeder per te_group. TE1/TE2 have distinct lr, and the seeder's median
                # anchor + gf_consol=lam/lr write BOTH use its single self.lr, so they cannot share
                # one. Anchor to te_lr + lam0_te=gf_consol*te_lr so the TE median lands at the TE's
                # TRUE dimensionless lam (not the UNet's ~30x-off one). Winner 2-fast TEs only.
                self._nsr_seeder_te = []
                for _gi, (_te_lyr, _te_lr) in enumerate(self.te_groups):
                    _win_te = [m for m in _te_lyr if getattr(m, "alpha_v_fast", 0.0) > 0.0
                               and hasattr(m, "arm_ratchet")]
                    if not _win_te:
                        continue
                    _sc_te = (_os.path.join(_os.path.dirname(_nsr_sc), f"concord_nsr_te{_gi}.json")
                              if _nsr_sc else None)
                    self._nsr_seeder_te.append(NoiseScaleSeeder(
                        _win_te, float(_te_lr), float(self.config.gf_consol) * float(_te_lr),
                        _win, sidecar=_sc_te, epoch_updates=_spe, per_row=_pr, now_t=self.step_idx))
        if getattr(self, "_nsr_seeder", None) is not None:
            self._nsr_seeder.step(self.step_idx)
        for _s in getattr(self, "_nsr_seeder_te", None) or []:
            _s.step(self.step_idx)
        # per-step refresh of the emb validity-scaled chase floors (exp55):
        # floor_row = clamp(floor0/rate, floor0, max(0.5, floor0)) from the CURRENT
        # scheduled scalar -- host fill on the armed device buffer (crosses the graph
        # boundary; the per-row variant was compiled at build). Before the first
        # seeding the rate is unknown -> broadcast floor0, identical to the scalar
        # path. The cap max(0.5, floor0) keeps the early schedule (floor0 ~ 0.9)
        # untouched and opens scaling room only once the floor decays below 0.5.
        if getattr(self, "_enr_prev", None):
            import prototype_packed_b as _pbb
            _f0 = float(_pbb._RATIO_CHASE_FLOOR)
            _fcap = max(0.5, _f0)
            for _tr in (getattr(self, "emb_trainables", None) or []):
                _c = getattr(_tr, "core", None)
                _rows = getattr(_c, "_chase_floor_rows", None) if _c is not None else None
                if _rows is None:
                    continue
                _rt = self._enr_rate.get(id(_tr))
                if _rt is None:
                    _rows.fill_(_f0)
                else:
                    torch.clamp(_f0 / _rt, min=_f0, max=_fcap, out=_rows)
        if self.autotuner is not None:
            # only ever the table tuner (DissipationAutoTuner); the hunting servos were removed
            self.autotuner.step(self.step_idx)
        # common-mode meter: once per epoch (servo-style once-per-t guard: before_step is
        # invoked `accum` times with the same update-step t).
        if (self.steps_per_epoch > 0 and self.step_idx > 0
                and self.step_idx % max(1, int(round(self.steps_per_epoch))) == 0
                and self.step_idx != self._cm_last_t):
            self._cm_last_t = self.step_idx
            self._emb_commonmode_meter()
            if getattr(self, "_enr_prev", None):
                self._emb_nsr_seed()
        # DECOUPLED floor decay (exp43, supersedes the shared-1-epoch D4). The CHASE floor protects
        # late-arriving young evidence, which keeps coming every epoch, so a floor that drops after
        # epoch 1 starves it (the lag-tax: high waste, low boil). It anneals over the WHOLE run
        # (winner_step's floor_horizon default). The LEAK floor must open the telescope fast, so it
        # keeps the ~1-epoch decay -- the old "like the telescope" rationale, right for the leak,
        # wrong for the chase. exp43 (staggered-onset): whole-run chase rescues the late cohort ~7x;
        # decoupling (leak 1ep) beats reverting BOTH to whole-run.
        _lh = self.steps_per_epoch if self.steps_per_epoch > 0 else None
        winner_step(self.step_idx, self.total_steps, self.layers, config=self.config,
                    leak_horizon=_lh)                # chase floor_horizon=None -> whole run
        # Adam bias-correction on the rank-1 v_hat: drive 1/(1-b2^t) into the per-device buffer the
        # kernel multiplies v_hat by (prototype :808) -- which is BEFORE both the preconditioner (:810)
        # and the cf=d_sv^2/v_hat block (:838), so this one driver fixes the cold-start step overshoot
        # AND the cf over-optimism (~125x at t=1 with the 1-epoch beta2). Keyed to the GLOBAL step_idx so
        # it -> 1.0 (inert) once warm and never re-warms on the per-epoch exit-42 relaunch.
        if self.config.bias_correct_v and self.layers:
            if getattr(self, "_bc_pb", None) is None:
                import prototype_packed_b as _pb
                _pb.set_bias_correct_v(True)
                for _m in self.layers:
                    _pb._v_bc_buf(_m.packed_w.device)          # pre-create so step 0 is already corrected
                self._bc_pb = _pb
                print("[concord] Adam v_hat bias-correction ON (1/(1-b2^t) driver; corrects step + cf; "
                      "vhat_warmstart suppressed)", flush=True)
            _b2 = float(getattr(self.layers[0], "adafactor_beta2", self.config.beta2))
            self._bc_pb.set_v_bias_correction(self._bc_pb.bias_correction_factor(self.step_idx, _b2))
        # Run-level infancy (dissipation_fill_ramp): write kappa_base * ramp into
        # the per-layer device buffers each step (the lr/sigma pattern; the host
        # mirror _gf_consol_value keeps the unmodulated base, so tuner commits and
        # this compose cleanly). Don't boil weight off while the pretrained mass
        # is in transit: friction ~0 early, 63% at the telescope time constant,
        # ~full by 3 tau, then the cosine lr takes it down -- rising-late, like
        # the fluctuation sigma. UNet + winner-recipe TEs; frozen-anchor TEs (alpha_v==0) pinned.
        if self.config.dissipation_fill_ramp:
            self._current_fill_ramp = self._fill_ramp(self.step_idx, self.config.alpha_v_fast)
            for m in self.layers:
                _rows = getattr(m, "_gf_consol_rows", None)
                if _rows is not None and _rows.numel() == m._gf_consol_buf.numel():
                    # per-row law (nsr_per_row): ramp the seeder's UNRAMPED row kappas
                    # instead of flat-filling -- a scalar fill_ here wiped the row
                    # structure every step and left the row law inert (audit C-4)
                    torch.mul(_rows, self._current_fill_ramp, out=m._gf_consol_buf)
                else:
                    m._gf_consol_buf.fill_(m._gf_consol_value * self._current_fill_ramp)
            for m in self.te_layers:               # winner TEs only; frozen anchors (alpha_v==0,
                if getattr(m, "alpha_v_fast", 0.0) > 0.0:   # gf_consol=0) have no fill-ramp to apply
                    # TE-row-servo: row-aware ramp (mirror the UNet branch) -- a scalar fill_ would
                    # flatten _gf_consol_rows every step (audit C-4). No-op path when single-fast
                    # (no _gf_consol_rows) -> the original scalar fill.
                    _rows = getattr(m, "_gf_consol_rows", None)
                    if _rows is not None and _rows.numel() == m._gf_consol_buf.numel():
                        torch.mul(_rows, self._current_fill_ramp, out=m._gf_consol_buf)
                    else:
                        m._gf_consol_buf.fill_(m._gf_consol_value * self._current_fill_ramp)
        # Secondary groups are SCHEDULE-ONLY (update_globals=False): sigma and the
        # ratio floors are module-global and the last winner_step writer wins for
        # the whole model -- the emb group's noise=False used to zero sigma for
        # the UNet too. The self.layers call above is the single global writer.
        for _te_lyr, _te_lr in self.te_groups:        # one schedule group per text encoder (own lr)
            winner_step(self.step_idx, self.total_steps, _te_lyr,
                        peak_lr=_te_lr, config=self.config, update_globals=False)
        # Quality-tag shield: refresh each plane's tag basis from the tags' current
        # deploy vectors (eager; the captured backward replays against the buffer).
        # No-op on planes without a shield wired.
        for tr in getattr(self, "emb_trainables", []):
            tr.update_quality_basis()
            # graph_te (Option C): refresh the group-shield projectors IN PLACE, eager, on the SAME
            # cadence + same pre-apply deploy as update_quality_basis (so the captured backward reads
            # this step's basis). Self-gates: no-op unless _use_capture_shield AND a group shield is
            # armed -> zero cost on the bridge path.
            tr._refresh_group_operators()
        if self.emb_cores:
            # embeddings get their OWN schedule group: warmup + cosine at the
            # embedding lr, fluctuation off (see register_embedding_cores), and
            # the release delay ("divot"): epoch 1 the UNet digs its basin
            # against the pristine anchors (lr=0 -> tick/evap/wd all zero; the
            # kernel still consolidates the packing residual); at release the
            # group gets a FRESH warmup on a shifted clock, cosine still ending
            # at the horizon, so the tokens take off gently into the prepared
            # basin instead of slaving to an untrained UNet's residual.
            clk = self._emb_clock(self.step_idx, self.total_steps, self.emb_delay_steps)
            if (clk is not None and self.emb_delay_steps > 0
                    and self.emb_auto_drive and not self._emb_drive_applied):
                self._finalize_embedding_calibration()   # first released step
            if clk is None:
                winner_step(0, max(1, self.total_steps), self.emb_cores, peak_lr=0.0,
                            noise=False, config=self.config, update_globals=False)
            else:
                winner_step(clk[0], clk[1], self.emb_cores, peak_lr=self.emb_lr,
                            noise=False, config=self.config, update_globals=False)
        # Per-layer LAMB trust-scale maintenance, AFTER every winner_step (so each layer's _lr_buf is
        # current): one alloc-free _lamb_scale_kernel/layer computes the scale from last step's norms +
        # zeroes the accumulators. HOST-SIDE + eager, one step stale -- stays OUT of the captured
        # backward, like lr/sigma/floors/fill-ramp. (The coh cf-discount needs no host prep: cf is
        # computed in-kernel from sig2/v_hat.)
        import prototype_packed_b as _ppb
        if _ppb._LAMB_TRUST:
            def _maint(m):
                if getattr(m, "_lr_buf", None) is not None:
                    _ppb._lamb_scale_kernel[(1,)](_ppb._lamb_wnorm_sq_buf(m.packed_w),
                                                  _ppb._lamb_stepnorm_sq_buf(m.packed_w),
                                                  m._lr_buf, _ppb._lamb_scale_buf(m.packed_w),
                                                  _ppb._LAMB_CAP, 1.0 / _ppb._LAMB_CLIP, _ppb._LAMB_CLIP)
            for m in self.layers:
                _maint(m)
            for m in self.te_layers + self.emb_cores:      # winner TEs + active embeddings; frozen
                if getattr(m, "alpha_v_fast", 0.0) > 0.0:  # anchors (alpha_v==0) don't step -> skip
                    _maint(m)

    @torch.no_grad()
    def after_step(self):
        """AFTER the optimizer update: gated rebalance (skips the no-op launches), tick."""
        self.gate()
        if self.arm_gate is not None:
            self.arm_gate()
        if getattr(self, "te_arm_gate", None) is not None:   # TE-row-servo: winner-TE arm-plane ratchet
            self.te_arm_gate()
        for _g in self.te_gates:                       # per-encoder gated rebalance
            _g()
        # A paired contrast step consolidates BOTH arms on the same image, so it
        # is TWO ticks (gradient events), not one: advance the clock by the step's
        # tick count so step_idx tracks ticks, not loader iterations. _contrast_step
        # sets _contrast_step_ticks=2; ordinary steps default to 1. RESET each update
        # so an ordinary step following a paired one does not inherit the 2. The
        # horizon + steps_per_epoch carry the matching (1+fraction) factor (set at
        # horizon-finalize), so the cosine still ends at the run's last epoch and
        # the epoch-keyed timescales stay calibrated. (accum==1 regime: after_step
        # fires per loader step; the contrast is guarded to accum==1.)
        _ticks = int(getattr(self, "_contrast_step_ticks", 1) or 1)
        self.step_idx += _ticks
        self._contrast_step_ticks = 1
        # Live washout/dissipation probe. Off-loop diagnostic; MUST never crash training
        # (try/except) and MUST not pressure VRAM at the ceiling (per-layer temps, freed
        # each iter; one host sync). Tune cadence / disable via CONCORD_HEALTH_EVERY (0=off).
        import os
        _hk = int(os.environ.get("CONCORD_HEALTH_EVERY", "50"))
        if _hk > 0 and self.layers and self.step_idx % _hk == 0:
            try:
                self._log_health()
            except Exception as _e:
                print(f"[concord-health] probe error (skipped): {_e!r}", flush=True)
        if os.environ.get("CONCORD_GRADW_DIAG"):
            try:
                from prototype_packed_b import read_gradw_diag
                _cos, _gn = read_gradw_diag()
            except Exception as _e:
                _cos, _gn = None, 0
                print(f"[concord-gradw] diag error (skipped): {_e!r}", flush=True)
            if _gn:
                self._gradw_count = getattr(self, "_gradw_count", 0) + 1
                print(f"[concord-gradw] step={self.step_idx} (#{self._gradw_count}) cos(.,DEPLOY) over "
                      f"{_gn} applies:  raw={_cos['raw']:+.4f}  step_live={_cos['step']:+.4f}  "
                      f"s_fast={_cos['sfast']:+.4f}  coh(sig)={_cos['coh']:+.4f}  "
                      f"incoh(noise)={_cos['incoh']:+.4f}", flush=True)
                _gmax = int(os.environ.get("CONCORD_GRADW_MAXSTEPS", "6"))
                if _gmax > 0 and self._gradw_count >= _gmax:
                    print("[concord-gradw] probe done -> exiting. The stage where cos first departs "
                          "~0 toward s_fast's -0.02 is where the contractive bias enters.", flush=True)
                    os._exit(0)

    @torch.no_grad()
    def _maybe_log_loss_ts(self, timestep, loss):
        """Buffer (batch timesteps, batch-mean loss) per step for the timestep-stratified loss
        analyzer (deconv_loss.py deconvolves per-band loss from these mixed-timestep observations).
        Opt-in via sentinel CONCORD_LOSS_TS.on. Stores DETACHED tensors (no per-step host sync) and
        batch-flushes to workspace/concord_loss_by_step.jsonl every 256 records. Meter-only; wrapped
        so a failure never disturbs training."""
        if not getattr(self, "_loss_ts_on", False):
            return
        try:
            buf = self._loss_ts_pending
            buf.append((timestep.detach().clone(),
                        loss.detach() if torch.is_tensor(loss) else float(loss)))
            if len(buf) >= 256:
                import os, json
                path = os.path.join(self.config.workspace_dir, "concord_loss_by_step.jsonl")
                with open(path, "a", encoding="utf-8") as f:
                    for ts, ls in buf:
                        f.write(json.dumps({"ts": ts.to("cpu").to(torch.int32).tolist(),
                                            "loss": round(float(ls), 6)}) + "\n")
                buf.clear()
        except Exception:
            self._loss_ts_pending = []      # drop the buffer on any error; never touch training

    @torch.no_grad()
    def log_console_snapshot(self, step, updates):
        """Per-BACKUP metrics for the across-backup trajectory console (meter-only,
        never affects training): per-layer velocity ||s_fast|| AND per-layer gate
        coherence (velocity-mass-weighted endorsed fraction, coh_vhat-matched to the
        live health line -- self.layers is UNet-only, so no frozen-TE coh~0 dilution),
        both in forward order, plus aggregate state-economy -- deploy / anchor /
        velocity norms, the leak gap, the coherence-ENDORSED velocity mass (now
        gate-accurate, so it agrees with the health line's coh), and the build-gate
        duty cycle as an exit-share proxy. Appends ONE json line per backup to
        workspace/concord_console_backuplog.jsonl; the forward-order layer names go
        in a one-time header line. Same no_grad packed-weight decode as _log_health
        (a sum-of-squares sweep, no eager forward), cheap at a backup boundary.
        Wrapped so a logging failure never aborts the backup. The trajectory panels
        (type velocity, state economy, layer strip) read this file; it fills in as
        the run backs up, since those metrics are NOT in the live health line."""
        import os, json
        try:
            if not self.layers:
                return
            import prototype_packed_b as _ppb
            path = os.path.join(self.config.workspace_dir, "concord_console_backuplog.jsonl")
            if not os.path.exists(path):
                names = [getattr(m, "_concord_name", f"layer{i}") for i, m in enumerate(self.layers)]
                with open(path, "w", encoding="utf-8") as f:
                    f.write(json.dumps({"header": True, "n": len(names), "names": names}) + "\n")
            dev = self.layers[0].packed_w.device
            acc = torch.zeros(6, dtype=torch.float64, device=dev)   # dep2 v2 s2 sf2 cohmass gap2
            build = torch.zeros(1, dtype=torch.float64, device=dev)
            ntot = 0
            vel = []
            coh_layer = []
            bmin = float(_ppb._EVAP_BUILD_MIN)
            for m in self.layers:
                p = m.packed_w
                sf = m.fine_sum() if hasattr(m, "fine_sum") else (p >> 16).float()
                ss = ((p << 16) >> 24).float()
                vs = ((p << 24) >> 24).float()
                sc = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - 15.0).exp2()
                sc128 = sc * 128.0
                sfW = sf * sc
                sf2 = sfW * sfW
                dep = (ss + vs) * sc128
                vel.append(round(float(sf2.sum().sqrt().item()), 5))     # per-layer ||s_fast||
                acc[0] += (dep * dep).sum(dtype=torch.float64)
                acc[1] += ((vs * sc128) ** 2).sum(dtype=torch.float64)
                acc[2] += ((ss * sc128) ** 2).sum(dtype=torch.float64)
                acc[3] += sf2.sum(dtype=torch.float64)
                acc[5] += (((ss - vs) * sc128) ** 2).sum(dtype=torch.float64)
                # Gate-accurate coherence, matching _log_health / the live health line (NOT the
                # raw sig/noise decode): raw sig from the leak gap, then UNDER coh_vhat discount
                # noise^2 by kappa/(cf+kappa), cf = d_sv_W^2/v_hat, exactly as the apply kernel does.
                # Meter-only. Per-layer coh = velocity-mass-weighted endorsed fraction (sums to coh_mass).
                C = float(getattr(m, "drift_cancel_C", 0.0))
                sig = C * (ss - vs) * 128.0
                noise2 = (sf - sig) ** 2
                if _ppb._USE_COH_VHAT and getattr(m, "alpha_v_fast", 0.0) > 0.0 \
                        and getattr(m, "v_row", None) is not None:
                    vh = m.v_row[:, None] * m.v_col[None, :] * m._sum_v_inv
                    vh = torch.maximum(vh, 0.03 * vh.mean())    # match kernel: floor v_hat at 3% of layer-mean
                    dsv_w = (ss - vs) * sc128
                    cf = (dsv_w * dsv_w) / (vh + 1e-30)
                    noise2 = noise2 * _ppb._COH_KAPPA / (cf + _ppb._COH_KAPPA)
                coh = (sig * sig) / (sig * sig + noise2 + 1e-30)
                sf2sum = sf2.sum(dtype=torch.float64)
                endorsed = (sf2 * coh).sum(dtype=torch.float64)
                acc[4] += endorsed                                     # endorsed (coherence-weighted) mass
                coh_layer.append(round(float(endorsed / (sf2sum + 1e-30)), 5))  # per-layer endorsed fraction
                if hasattr(m, "arm_row_exp"):                          # 2-fast build gate (arm_shift guard)
                    arm_e = (m.arm_row_exp.to(torch.float32)[:, None]
                             + m.arm_col_exp.to(torch.float32)[None, :])
                    fine_raw = sf / torch.pow(2.0, arm_e)
                    pb = (fine_raw.abs() / ((7.0 - arm_e).clamp(min=0.0) + 1e-30)).clamp(max=1.0)
                else:
                    pb = (sf.abs() / (bmin + 1e-30)).clamp(max=1.0)
                build += pb.sum(dtype=torch.float64)
                ntot += sf.numel()
            a = acc.tolist()
            rec = {"step": int(step), "updates": int(updates),
                   "deploy": round(a[0] ** 0.5, 4), "vslow": round(a[1] ** 0.5, 4),
                   "sslow": round(a[2] ** 0.5, 4), "sfast": round(a[3] ** 0.5, 5),
                   "coh_mass": round(a[4] ** 0.5, 5), "gap": round(a[5] ** 0.5, 4),
                   "build_ok": round(100.0 * float(build.item()) / max(1, ntot), 3),
                   "vel": vel, "coh": coh_layer}
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            # ---- claim-1 flattening tracker (opt-in: sentinel CONCORD_LAYER_PR.on at the repo root) ----
            # Per-TE-layer output-channel-energy PARTICIPATION RATIO (effective channel count) of the
            # DEPLOY weight -- a VRAM-safe, SVD-FREE proxy for representation flattening (rising PR =
            # output energy spreading off the dominant axes = more directions carried). Winner TEs
            # (alpha_v>0) only; frozen anchors don't move. Own file (console reader untouched); inside
            # this try, so a failure never aborts a backup. Same per-layer packed decode as above.
            _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            if self.te_layers and os.path.exists(os.path.join(_root, "CONCORD_LAYER_PR.on")):
                pr_path = os.path.join(self.config.workspace_dir, "concord_layer_pr.jsonl")
                if not os.path.exists(pr_path):
                    pn = [getattr(m, "_concord_name", f"te{i}") for i, m in enumerate(self.te_layers)]
                    with open(pr_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps({"header": True, "n": len(pn), "names": pn}) + "\n")
                prs = []
                for m in self.te_layers:
                    p = m.packed_w
                    ss = ((p << 16) >> 24).float()
                    vs = ((p << 24) >> 24).float()
                    sc128 = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - 15.0).exp2() * 128.0
                    dep = (ss + vs) * sc128                    # deploy weight [out, in]
                    e = (dep * dep).sum(dim=1)                 # per-output-channel energy [out]
                    prs.append(round(float((e.sum() ** 2) / (e.pow(2).sum() + 1e-30)), 3))
                with open(pr_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": int(step), "updates": int(updates), "pr": prs}) + "\n")
        except Exception as e:
            print(f"[concord] console snapshot skipped: {e}", flush=True)

    @torch.no_grad()
    def _log_health(self):
        """Per-probe intrinsic state from the live packed weights (same decode as
        measure_coherence): position/anchor/velocity norms, whether evaporation is firing
        (build_ok), the coherence the gate reads (incl. the coh_vhat cf-modulation when on, so it
        matches the kernel's ACTUAL gate coh, NOT measure_coherence's raw sig/noise decode), and whether the velocity pulls the
        position toward zero (cos<0). Decoded on-device per layer, accumulated in fp64
        device scalars, synced once. Watch: ||deploy|| (washout), build_ok (0=dissipation
        inert), coh (~0.98 healthy), cos(sf,dep) (<0 contractive)."""
        # Decode one layer-set into fp64 aggregates, accumulated on-device, synced once.
        # acc = [dep, v, s, sf, sf*coh, dot, build, gap]. gap = ||(s_slow-v_slow)*128|| in
        # weight units -- the leak lag the coherence is built from (sig = C*gap). For the
        # creep TE it starts ~0 (gap-zero init) and grows as v_slow lags s_slow; for the
        # frozen TE (s_slow=0, v_slow=W) it sits at ~||W|| and is static.
        def _metrics(layers):
            # build_ok = the build gate's EXPECTED (stochastic) firing rate -- the real duty cycle, NOT a
            # >= threshold count (which reads ~0 whenever the chase pins the accumulator below the scale,
            # hiding whether the drain fires at all). The gate DIFFERS by kernel, so each layer is matched
            # to its own: the 2-fast arms use the arm_shift guard p_build = min(|fine|/max(7-(ar+ac),0), 1)
            # with fine = RAW e_L+e_H; the winner 'b' kernel uses p_build = min(|s_fast|/evap_build_min, 1).
            # Reporting evap_build_min on the int8 arm plane reads a structural ~1% regardless of the real
            # gate (|fine|<=64 < 128) -- the exact build_ok~0 pathology the arm_shift guard was built to fix.
            import prototype_packed_b as _ppb
            build_min = float(_ppb._EVAP_BUILD_MIN)
            dev = layers[0].packed_w.device
            acc = torch.zeros(8, dtype=torch.float64, device=dev)
            ntot = 0
            for m in layers:
                p = m.packed_w
                if hasattr(m, 'fine_sum'):
                    sf = m.fine_sum()   # 2-fast: fine units via the accumulator scale
                else:
                    sf = (p >> 16).float()
                ss = ((p << 16) >> 24).float()
                vs = ((p << 24) >> 24).float()
                sc = (m.row_exp.float()[:, None] + m.col_exp.float()[None, :] - 15.0).exp2()
                sc128 = sc * 128.0
                sfW = sf * sc
                dep = (ss + vs) * sc128
                acc[0] += (dep * dep).sum(dtype=torch.float64)
                acc[1] += ((vs * sc128) ** 2).sum(dtype=torch.float64)
                acc[2] += ((ss * sc128) ** 2).sum(dtype=torch.float64)
                acc[5] += (sfW * dep).sum(dtype=torch.float64)
                acc[7] += (((ss - vs) * sc128) ** 2).sum(dtype=torch.float64)
                del dep
                C = float(getattr(m, "drift_cancel_C", 0.0))
                sig = C * (ss - vs) * 128.0
                noise = sf - sig
                noise2 = noise * noise
                if _ppb._USE_COH_VHAT and getattr(m, "alpha_v_fast", 0.0) > 0.0 \
                        and getattr(m, "v_row", None) is not None:
                    # Match the gate's ACTUAL coh under coh_vhat: discount noise^2 by kappa/(cf+kappa)
                    # with cf = d_sv_W^2/v_hat, exactly as the apply kernel does (scale_fwd cancels in the
                    # coh ratio, so the dimensionless discount applies straight to the mantissa noise^2).
                    vh = m.v_row[:, None] * m.v_col[None, :] * m._sum_v_inv
                    vh = torch.maximum(vh, 0.03 * vh.mean())    # match kernel: floor v_hat at 3% of layer-mean
                    dsv_w = (ss - vs) * sc128
                    cf = (dsv_w * dsv_w) / (vh + 1e-30)
                    noise2 = noise2 * _ppb._COH_KAPPA / (cf + _ppb._COH_KAPPA)
                coh = (sig * sig) / (sig * sig + noise2 + 1e-30)
                sf2 = sfW * sfW
                acc[3] += sf2.sum(dtype=torch.float64)
                acc[4] += (sf2 * coh).sum(dtype=torch.float64)
                if hasattr(m, 'arm_row_exp'):     # 2-fast: the arm_shift guard IS the build gate the kernel applies
                    arm_e = (m.arm_row_exp.to(torch.float32)[:, None]
                             + m.arm_col_exp.to(torch.float32)[None, :])
                    fine_raw = sf / torch.pow(2.0, arm_e)          # sf = (e_L+e_H)*2^arm_e -> RAW counts (kernel's `fine`)
                    arm_shift = (7.0 - arm_e).clamp(min=0.0)
                    p_build = (fine_raw.abs() / (arm_shift + 1e-30)).clamp(max=1.0)
                else:                             # winner / 'b' kernel: evap_build_min gate
                    p_build = (sf.abs() / (build_min + 1e-30)).clamp(max=1.0)
                acc[6] += p_build.sum(dtype=torch.float64)
                ntot += sf.numel()
            return acc.tolist(), ntot

        def _line(tag, layers):
            if not layers:
                return
            a, ntot = _metrics(layers)
            nd, nsf = a[0] ** 0.5, a[3] ** 0.5
            # realized-coherence quantiles (the tau-calibration meter,
            # docs/COH_TIMESTEP_CALIBRATION.md): the gate input's own distribution.
            # q90 is the compression ceiling; (1-q50)/tau is the median commit deficit
            # in REAL units; read-and-zero -> the window since the last health line.
            _cq = ""
            if tag == "unet":
                try:
                    from prototype_packed_2fast import read_cohq
                    _r = read_cohq(layers[0].packed_w.device) if layers else None
                    if _r is not None:
                        _cq = f" cohq={_r[0]:.2f}/{_r[1]:.2f}/{_r[2]:.2f}"
                except Exception:
                    pass
            print(f"[concord-health:{tag}] step={self.step_idx} "
                  f"||deploy||={nd:.1f} ||v||={a[1] ** 0.5:.1f} ||s||={a[2] ** 0.5:.1f} "
                  f"||s_fast||={nsf:.3f} build_ok={100.0 * a[6] / max(1, ntot):.3f}% "
                  f"coh={a[4] / max(a[3], 1e-30):.3f} cos(sf,dep)={a[5] / max(nsf * nd, 1e-30):+.3f} "
                  f"gap={a[7] ** 0.5:.1f}{_cq}",
                  flush=True)

        _line("unet", self.layers)
        _line("te", self.te_layers)   # frozen TE -> coh~0, gap~||W||; creep TE -> coh live, gap grows from ~0
        # Prequential servo meter readout: aggregate window cosine since the last health line.
        # Sign law: s < 0 => lower kappa would reduce in-stream loss, s > 0 => raise (exp 24 --
        # consume only with a margin; the in-stream signal is ~30% attenuated).
        _pq = [m for m in self.layers
               if getattr(m, '_preq_meter', None) is not None]
        if _pq:
            tot = torch.stack([_m._preq_meter for _m in _pq]).sum(0)
            # The meter is PASSIVE (no controller consumes it), so _log_health owns the zero
            # (else the accumulator grows all run and the delta readout loses float precision).
            sd, sg, sa = float(tot[0]), float(tot[1]), float(tot[2])
            for _m in _pq:
                _m._preq_meter.zero_()
            s = sd / max(1e-30, (max(0.0, sg) * max(0.0, sa)) ** 0.5)
            print(f"[concord-preq:unet] step={self.step_idx} s={s:+.5f} "
                  f"(window cosine over {len(_pq)} layers since last "
                  f"health line)", flush=True)

    @torch.no_grad()
    def consolidate_into_unet(self, unet):
        """DEPLOY: replace the packed Concord layers in-place with standard nn.Linear /
        nn.Conv2d holding the CONSOLIDATED weights (drops the transient s_fast), so the
        UNet saves and loads as an ordinary SDXL UNet. Destructive -- call once before the
        FINAL save; do not keep Concord-training after."""
        import torch.nn as nn
        from prototype_packed_b import ConcordConv2dPackedB, ConcordLinearPackedB
        n = 0
        for parent in unet.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, ConcordConv2dPackedB):     # subclass -> check first
                    w = child.consolidated_weight().reshape(
                        child.out_channels, child.in_channels, child.kh, child.kw)
                    new = nn.Conv2d(child.in_channels, child.out_channels, (child.kh, child.kw),
                                    stride=child.stride, padding=child.padding,
                                    bias=child.bias is not None)
                elif isinstance(child, ConcordLinearPackedB):
                    w = child.consolidated_weight()             # [out, in]
                    new = nn.Linear(child.in_features, child.out_features,
                                    bias=child.bias is not None)
                else:
                    continue
                new = new.to(device=child.packed_w.device, dtype=w.dtype)
                new.weight.data.copy_(w)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.detach().to(new.bias.dtype))
                setattr(parent, name, new)
                n += 1
        print(f"[concord] consolidated {n} layers -> standard nn.Linear/nn.Conv2d for deploy")
        return n

    @torch.no_grad()
    def materialize_unet_deploy(self):
        """Reversible deploy window, ZERO GPU allocations: D2H-copy each packed
        word to CPU (no device temp), extract/stash the s_fast bits CPU-side,
        mask s_fast out of packed_w IN PLACE (the deploy weight is exactly the
        low 16 bits). Fused mode then dequantizes the deploy weight directly;
        cached mode re-materializes its existing buffer. Restore ORs the bits
        back through ONE reused device scratch — no per-layer allocation churn
        (sampling's VRAM fragmentation is the documented wedge; do not feed it)."""
        import os as _os
        import prototype_packed_b as ppb
        stash = []
        _evict_env = _os.environ.get("CONCORD_SAMPLE_EVICT_PACKED", "1")
        evict = _evict_env != "0"
        _al0 = torch.cuda.memory_allocated() / 2 ** 30
        self._evict_fused_was = None
        # embedding cores included: their forward gathers reconstruct from
        # packed_w directly, so masking alone makes sampling deploy-true.
        # Embedding cores are NEVER evicted (their sampler forward reads
        # packed rows directly and they are small).
        for m in self.layers + self.emb_cores:
            pk_cpu = m.packed_w.detach().to("cpu")          # D2H, no device temp
            stash.append(((pk_cpu >> 16).to(torch.int16)))  # CPU-side extract
            m.packed_w &= 0xFFFF                            # in place
            if evict and m in self.layers:
                # LAYER-BY-LAYER eviction: materialize this layer's
                # deploy bf16 (~tens of MB), then hand its packed word to
                # the host -- device usage decreases monotonically, no
                # allocation peak. bf16 deploy total (~5.4G) replaces the
                # int32 packed plane (~10.9G) for the whole window.
                wbuf = getattr(m, '_bf16_weight_buf', None)
                if wbuf is None or wbuf.shape != m.packed_w.shape:
                    wbuf = torch.empty(m.packed_w.shape, dtype=torch.bfloat16,
                                       device=m.packed_w.device)
                    m._bf16_weight_buf = wbuf
                if hasattr(m, 'arm_row_exp'):
                    import prototype_packed_2fast as p2f
                    p2f.materialize_packed2_bf16(
                        m.packed_w, m.row_exp, m.col_exp,
                        m.arm_row_exp, m.arm_col_exp, out=wbuf,
                        mantissa_bias=m.MANTISSA_BIAS)
                else:
                    ppb.materialize_packed_bf16(m.packed_w, m.row_exp,
                                                m.col_exp, out=wbuf,
                                                mantissa_bias=m.MANTISSA_BIAS)
                # full word to host; shape preserved for _ensure_buffers.
                old_dev = m.packed_w          # the on-device word (post-mask)
                # DE-REGISTER packed_w as a buffer for the window. The sampler
                # moves the UNet back to the train device for its forward
                # (model.to(temp) then the sampler's own .to(train_device)),
                # and .to() copies every REGISTERED BUFFER -- which re-
                # materialized the packed plane on device right after eviction
                # moved it off (GPU-verified: census int32-2D reappears ==
                # exactly 2x the bf16 deploy cache). A PLAIN attribute is
                # invisible to .to() -- the same reason _bf16_weight_buf
                # survives as one. Restore re-registers it on device.
                if 'packed_w' in m._buffers:
                    del m._buffers['packed_w']
                m.packed_w = pk_cpu           # PLAIN attr (host); .to() ignores it
                # FORCE-FREE the old device bytes now, even though the last
                # training step's autograd ctx still pins them (ctx.args holds
                # packed_w until its graph drops). resize_(0) frees the
                # allocation regardless of the stale ref; safe here -- the
                # captured graph is released before this runs (no replay reads
                # the recorded address) and sampling is no_grad.
                try:
                    old_dev.untyped_storage().resize_(0)
                except Exception:
                    pass                      # pre-2.x fallback: refcount drop
                del old_dev
                m._evicted = True
                continue
            if not ppb._FUSED_MATMUL:
                wbuf, _, _ = m._ensure_buffers()
                if hasattr(m, 'arm_row_exp'):
                    import prototype_packed_2fast as p2f
                    p2f.materialize_packed2_bf16(
                        m.packed_w, m.row_exp, m.col_exp,
                        m.arm_row_exp, m.arm_col_exp, out=wbuf,
                        mantissa_bias=m.MANTISSA_BIAS)
                else:
                    ppb.materialize_packed_bf16(m.packed_w, m.row_exp, m.col_exp,
                                                out=wbuf,
                                                mantissa_bias=m.MANTISSA_BIAS)
        if evict and any(getattr(m, '_evicted', False) for m in self.layers):
            # window-wide: forwards must read the bf16 buffers, not dequant
            # from (now absent) packed words. Both kernel families read
            # this module flag at call time; restored on exit.
            self._evict_fused_was = ppb._FUSED_MATMUL
            ppb._FUSED_MATMUL = False
        # Self-report the window: turns "why didn't packed clear?" from
        # inference-off-the-census into a stated fact. n_evicted==0 with
        # evict=False => the env var disabled it; evicted>0 but alloc still
        # high => a genuine pin. Cheap (one host sync); on unless GRAPHMEM off.
        if _os.environ.get("CONCORD_GRAPHMEM", "1").strip().lower() not in (
                "0", "false", "no", "off", ""):
            _al1 = torch.cuda.memory_allocated() / 2 ** 30
            _nev = sum(1 for m in self.layers if getattr(m, '_evicted', False))
            print(f"[concord-evict] CONCORD_SAMPLE_EVICT_PACKED={_evict_env} "
                  f"evict={evict} fused_now={ppb._FUSED_MATMUL} "
                  f"layers={len(self.layers)} evicted={_nev}/{len(self.layers)} "
                  f"alloc {_al0:.2f}G -> {_al1:.2f}G (freed {_al0 - _al1:+.2f}G)",
                  flush=True)
        return stash

    @torch.no_grad()
    def restore_unet_deploy(self, stash):
        import prototype_packed_b as ppb
        if getattr(self, '_evict_fused_was', None) is not None:
            ppb._FUSED_MATMUL = self._evict_fused_was
            self._evict_fused_was = None
        for m in self.layers:
            if getattr(m, '_evicted', False):
                dev = m.row_exp.device        # the layer's live device
                # RE-REGISTER as a device buffer (eviction de-registered it so
                # the sampler's device moves could not drag it back on).
                _host = m.packed_w
                if 'packed_w' not in m._buffers and hasattr(m, 'packed_w'):
                    del m.packed_w            # drop the plain attr first
                m.register_buffer('packed_w', _host.to(dev))
                m._evicted = False
                # packed_w was reallocated: data_ptr-keyed registries are
                # stale. Re-register per-layer meters or the next
                # recapture routes them to the shared sinks.
                if getattr(m, '_boil_meter', None) is not None:
                    ppb.register_layer_meters(m.packed_w, m._boil_meter,
                                              m._memgap_meter)
                if getattr(m, '_preq_meter', None) is not None:
                    import prototype_packed_2fast as _p2f
                    _p2f.register_preq_meter(m.packed_w, m._preq_meter)
                if getattr(m, '_nsr_buf', None) is not None:
                    import prototype_packed_2fast as _p2f
                    _p2f.register_nsr_meter(m.packed_w, m._nsr_buf, m._nsr_idx)
                if ppb._FUSED_MATMUL:
                    # fused training never reads the per-layer bf16 copy;
                    # keeping it would silently re-add the ~5.4G the
                    # window existed to save. Free it.
                    m._bf16_weight_buf = None
                if not ppb._FUSED_MATMUL:
                    # cached mode: refresh the LIVE weight into the buffer
                    wbuf, _, _ = m._ensure_buffers()
                    if hasattr(m, 'arm_row_exp'):
                        import prototype_packed_2fast as p2f
                        p2f.materialize_packed2_bf16(
                            m.packed_w, m.row_exp, m.col_exp,
                            m.arm_row_exp, m.arm_col_exp, out=wbuf,
                            mantissa_bias=m.MANTISSA_BIAS)
                    else:
                        ppb.materialize_packed_bf16(
                            m.packed_w, m.row_exp, m.col_exp, out=wbuf,
                            mantissa_bias=m.MANTISSA_BIAS)
        targets = self.layers + self.emb_cores
        dev = targets[0].packed_w.device if targets else "cuda"
        scratch = getattr(self, "_deploy_scratch", None)
        need = max((m.packed_w.numel() for m in targets), default=0)
        if scratch is None or scratch.numel() < need:
            scratch = torch.empty(need, dtype=torch.int32, device=dev)
            self._deploy_scratch = scratch                   # reused across samples
        for m, sf in zip(targets, stash):
            word = (sf.to(torch.int32) << 16)                # CPU-side
            n = m.packed_w.numel()
            if scratch.device == m.packed_w.device and scratch.numel() >= n:
                scratch[:n].copy_(word.reshape(-1))          # one H2D, no alloc (normal path)
                m.packed_w |= scratch[:n].view_as(m.packed_w)
            else:
                # MIXED-DEVICE model: a KeyboardInterrupt mid-sample unwinds while the
                # offload conductor has layers split across cpu/cuda, so the shared
                # scratch (sized on targets[0]'s device) mismatches later layers. Fall
                # back to a direct per-layer transfer -- alloc churn is fine on the
                # crash-unwind path, and losing this restore would permanently drop the
                # layer's s_fast. (Observed 2026-07-02: restore raised cuda-vs-cpu inside
                # KI handling, replacing the clean stop with exit code 1.)
                m.packed_w |= word.reshape(-1).to(m.packed_w.device).view_as(m.packed_w)
            if not ppb._FUSED_MATMUL:
                wbuf, _, _ = m._ensure_buffers()
                if hasattr(m, 'arm_row_exp'):
                    import prototype_packed_2fast as p2f
                    p2f.materialize_packed2_bf16(
                        m.packed_w, m.row_exp, m.col_exp,
                        m.arm_row_exp, m.arm_col_exp, out=wbuf,
                        mantissa_bias=m.MANTISSA_BIAS)
                    continue
                ppb.materialize_packed_bf16(m.packed_w, m.row_exp, m.col_exp,
                                            out=wbuf,
                                            mantissa_bias=m.MANTISSA_BIAS)


    def materialize_te_deploy(self):
        """REVERSIBLE TE deploy: replace each text-encoder ConcordLinearPackedB with a temp
        nn.Linear holding its consolidated_weight() (DROPS s_fast, matching the UNet deploy --
        the TE deploys its consolidated position, not the live transient; this keeps the
        coherence-gated s_fast evaporation from showing up directly in the deployed TE weight).
        Returns a stash; pass it to restore_te_deploy() in a finally so training continues."""
        import torch.nn as nn
        from prototype_packed_b import ConcordLinearPackedB
        if not self.te_layers or not self.te_encoders:
            return []
        # ONLY the swapped TE transformer Linears (te_layers), across every anchored encoder. The
        # control-plane embedding core is also a ConcordLinearPackedB living inside the TE module
        # tree, but it has its own save bridge (materialize_packed_embeddings_to_vectors) -> must
        # NOT be clobbered.
        te_set = {id(m) for m in self.te_layers}
        stash = []
        for te in self.te_encoders:
            for parent in te.modules():
                for name, child in list(parent.named_children()):
                    if isinstance(child, ConcordLinearPackedB) and id(child) in te_set:
                        w = child.consolidated_weight()   # drop s_fast: deploy the consolidated position (like the UNet)
                        lin = nn.Linear(child.in_features, child.out_features,
                                        bias=child.bias is not None).to(
                            device=child.packed_w.device, dtype=w.dtype)
                        lin.weight.data.copy_(w)
                        if child.bias is not None:
                            lin.bias.data.copy_(child.bias.detach().to(lin.bias.dtype))
                        setattr(parent, name, lin)
                        stash.append((parent, name, child))
        return stash

    @torch.no_grad()
    def restore_te_deploy(self, stash):
        """Undo materialize_te_deploy: put the packed ConcordLinearPackedB modules back so
        training continues with the exact pre-save state (incl. the frozen v_slow anchor)."""
        for parent, name, packed in stash:
            setattr(parent, name, packed)


# ---------------------------------------------------------------------------
# Norm-preserving packed embeddings (ControlPlaneEmbedding + ConcordPackedEmbedding).
#
# Replaces the plain-SGD AdditionalEmbeddingWrapper path for the TRAINABLE new tokens:
# each token becomes a row of a per-TE ConcordPackedEmbedding that self-steps INSIDE the
# captured backward and pins its DEPLOY norm to the vocab median -- the regularization the
# plain-SGD path lacked (embedding_learning_rate=1e-3 + no norm clamp -> overfit). The
# control plane REPLACES the TE's token_embedding (capture-safe, branch-free forward).
#
# OneTrainer's save/load is preserved: at save time the packed deploy vectors are
# materialized back into embedding.*.vector (-> standard clip_l/clip_g safetensors) and the
# ORIGINAL token_embedding is temporarily restored so the TE serializes as a plain
# CLIPTextModel. On resume _setup_embeddings restores .vector from the backup and
# setup_packed_embeddings re-packs it -> the trained tokens round-trip.
# ---------------------------------------------------------------------------

def packed_embeddings_active(config) -> bool:
    """True when the trainable new-token embeddings should route through the packed
    self-stepping core instead of plain SGD: Concord optimizer + flag on + something
    (non-output) to train."""
    from modules.util.enum.Optimizer import Optimizer, is_concord_family
    return (is_concord_family(config.optimizer.optimizer)
            and bool(getattr(config, "concord_packed_embeddings", False))
            and (config.train_any_embedding()
                 or bool(getattr(config, "concord_train_caption_vocab", False))))


def _packed_trainable_uuids(config):
    """uuids of the additional embeddings that train via the packed core (train=True,
    non-output)."""
    out = set()
    for ec in config.all_embedding_configs():
        if getattr(ec, "train", False) and not getattr(ec, "is_output_embedding", False):
            out.add(ec.uuid)
    return out


def collect_caption_token_ids(config, tokenizer):
    """Static union of base-vocab token ids that appear in any TRAINING caption, for this tokenizer.
    Scanned once at setup (the attach set is static). add_special_tokens=False, no truncation (the
    sighting gate makes a never-sighted row free), special ids + OOV stripped. Robust to concepts
    being dicts (json.load) or ConceptConfig objects."""
    import os
    def _g(o, k, d=None):
        return o.get(k, d) if isinstance(o, dict) else getattr(o, k, d)
    try:
        from modules.util import path_util
        img_exts = set(path_util.supported_image_extensions())
    except Exception:
        img_exts = {".jpg", ".jpeg", ".jpe", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or 0)
    from collections import Counter
    counts = Counter()

    def list_images(root, recurse):
        out = []
        try:
            names = os.listdir(root)
        except OSError:
            return out
        for name in names:
            p = os.path.join(root, name)
            if os.path.isdir(p):
                if recurse and not name.startswith("."):
                    out += list_images(p, recurse)
            else:
                stem, ext = os.path.splitext(p)
                if (ext.lower() in img_exts
                        and not (stem.endswith("-masklabel") or stem.endswith("-condlabel"))):
                    out.append(p)
        return out

    def caption_lines(img, tcfg):
        ps = _g(tcfg, "prompt_source", "sample")
        src = str(getattr(ps, "value", ps) or "sample").lower()   # enum (ConceptConfig obj) or str (dict)
        if src == "filename":
            return [os.path.splitext(os.path.basename(img))[0]]
        if src == "concept":
            pp = _g(tcfg, "prompt_path", "") or ""
            if pp and os.path.exists(pp):
                with open(pp, encoding="utf-8") as f:
                    return [ln.strip() for ln in f if ln.strip()]
            return []
        txt = os.path.splitext(img)[0] + ".txt"        # 'sample': per-image sidecar, one caption per line
        if os.path.exists(txt):
            with open(txt, encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        return []

    concepts = getattr(config, "concepts", None)
    if not concepts:                       # CLI / eager-GUI: concepts load lazily AFTER setup -> load the file now
        cfn = getattr(config, "concept_file_name", None)
        if cfn and os.path.exists(cfn):
            try:
                import json
                from modules.util.config.ConceptConfig import ConceptConfig
                with open(cfn, "r", encoding="utf-8") as f:
                    concepts = [ConceptConfig.default_values().from_dict(c) for c in json.load(f)]
            except Exception as _e:
                print(f"[concord] caption-vocab: could not load concept file {cfn} ({_e})")
                concepts = []
    if not concepts:
        print("[concord] caption-vocab: WARNING -- no concepts resolved; attaching NO caption tokens")
    for concept in (concepts or []):
        if not _g(concept, "enabled", True):
            continue
        if "VALIDATION" in str(_g(concept, "type", "") or "").upper():
            continue
        root = _g(concept, "path", "") or ""
        if not root or not os.path.isdir(root):
            continue
        recurse = bool(_g(concept, "include_subdirectories", False))
        tcfg = _g(concept, "text", {}) or {}
        for img in list_images(root, recurse):
            for line in caption_lines(img, tcfg):
                toks = tokenizer(line, add_special_tokens=False, truncation=False).input_ids
                counts.update(int(t) for t in toks)
    ids = set(counts)
    ids -= special
    if vocab_size > 0:
        ids = {t for t in ids if t < vocab_size}
    # (b) CONTENT FILTER: separate real content from function-word "glue" (the, of, and,
    # punctuation). content_only drops whole-word STOPWORDS + pure punctuation/digits, but
    # KEEPS sub-word fragments (theophite -> the+oph+ite compose a named concept) and
    # content words. min_count drops incidental low-frequency tokens. The norm artifact is
    # a SEPARATE fix (concord_embedding_preserve_norm = don't rescale), not this.
    content_only = bool(getattr(config, "concord_caption_vocab_content_only", True))
    min_count = int(getattr(config, "concord_caption_vocab_min_count", 1) or 1)
    if content_only or min_count > 1:
        import re as _re
        _STOP = frozenset((
            "a an the and or but nor so than of to in on at for with by from as into onto "
            "over under about above below between through during before after near behind "
            "it its he she they them we you i me my your his her their our this that these "
            "those who whom which what is are was were be been being am has have had do does "
            "did will would can could should may might must not no then there here up down "
            "out off again once very too just also only own same such all any both each more "
            "most some few").split())
        n0 = len(ids)
        inv = {v: k for k, v in tokenizer.get_vocab().items()}
        def _keep(t):
            if counts[t] < min_count:
                return False
            if content_only:
                s = inv.get(t, "")
                body = s[:-4] if s.endswith("</w>") else s
                if not _re.search(r"[^\W\d_]", body):
                    return False                              # pure punctuation / digits
                if s.endswith("</w>") and body.lower() in _STOP:
                    return False                              # whole-word function glue
            return True                                       # sub-word fragments + content KEPT
        ids = {t for t in ids if _keep(t)}
        print(f"[concord] caption-vocab filter: {n0} -> {len(ids)} tokens "
              f"(content_only={content_only}, min_count={min_count})")
    return ids


def setup_packed_embeddings(model, config):
    """Replace each SDXL text encoder's token_embedding with a ControlPlaneEmbedding whose
    trainable rows are a norm-preserving ConcordPackedEmbedding. Call AFTER _setup_embeddings
    (so .vector is restored/created + the placeholder tokens are in the tokenizer) and AFTER
    the fused-matmul flag is set (the packed core reads _FUSED_MATMUL at construction).
    Stores model.concord_control_planes = [{te_idx, te, cp, base, row_map}, ...]."""
    import sys
    from pathlib import Path
    cdir = str((Path(__file__).parent / "concord").resolve())
    if cdir not in sys.path:
        sys.path.insert(0, cdir)
    from control_plane import ControlPlaneEmbedding

    train_uuids = _packed_trainable_uuids(config)
    lr = float(config.embedding_learning_rate or config.learning_rate)   # emb LR is nullable (=use base LR)
    specs = [
        (1, model.text_encoder_1, model.tokenizer_1, model.all_text_encoder_1_embeddings()),
        (2, model.text_encoder_2, model.tokenizer_2, model.all_text_encoder_2_embeddings()),
    ]
    # Quality-tag shield (optional): mark some trainable embeddings as low-quality
    # "tags" (by placeholder). They train FREELY as the defect sink (a droppable
    # knob); every OTHER trainable embedding's gradient is projected off the tags'
    # subspace so it learns subject content from a bad image but not its badness.
    # Resolve the config once; partition rows per plane inside the loop.
    q_tags_csv = (getattr(config, "concord_embedding_quality_tags", "") or "")
    q_on = (bool(getattr(config, "concord_embedding_quality_orthogonal", False))
            and bool(q_tags_csv.strip()))
    q_one_sided = (str(getattr(config, "concord_embedding_quality_mode", "hard"))
                   .strip().lower() == "one_sided")
    q_tag_set = {s.strip() for s in q_tags_csv.replace("\n", ",").split(",") if s.strip()}
    s_tags_csv = (getattr(config, "concord_embedding_style_tags", "") or "")
    s_tag_set = {s.strip() for s in s_tags_csv.replace("\n", ",").split(",") if s.strip()}
    s_overlap = q_tag_set & s_tag_set
    if s_overlap:
        # a token cannot be both sinks: quality wins the tag role
        print(f"[concord] style-tag shield: {sorted(s_overlap)} are already quality "
              f"tags -> removed from the style list (quality wins)", flush=True)
        s_tag_set -= s_overlap
    planes = []
    for te_idx, te, tokenizer, embeddings in specs:
        base = te.text_model.embeddings.token_embedding
        if isinstance(base, ControlPlaneEmbedding):          # re-setup on a persisted model
            base = base.base
            te.text_model.embeddings.token_embedding = base
        cp = ControlPlaneEmbedding(base)
        median = base.weight.float().norm(dim=1).median().item()
        tids, inits, row_map = [], [], []
        for emb in embeddings:
            if emb.uuid not in train_uuids:
                continue
            ids = tokenizer.convert_tokens_to_ids(emb.text_tokens)   # token_count ids
            for k, tid in enumerate(ids):
                tids.append(int(tid))
                inits.append(emb.vector[k].detach().float())
                row_map.append((emb, k))
        # Caption-vocab: also train the BASE-vocab tokens that appear in the dataset captions, via the
        # same single attach_trainable (it OVERWRITES cp.trainable, so caption rows must extend the same
        # tids/inits/row_map). Seed from base.weight[tid] (base tokens have no emb.vector); mark each with
        # a (None, tid) sentinel so the save bridge writes deploy back to base.weight[tid].
        caption_tids = []
        if bool(getattr(config, "concord_train_caption_vocab", False)):
            cap_ids = collect_caption_token_ids(config, tokenizer)
            cap_ids -= set(tids)                              # de-dup vs added-token ids
            _san = getattr(model, "concord_sanitize", None)  # don't resurrect a sanitized (zeroed) token
            if _san is not None:
                cap_ids -= set(getattr(_san, "ids1" if te_idx == 1 else "ids2", None) or [])
            for tid in sorted(cap_ids):
                if tid < int(cp.kind.shape[0]) and int(cp.kind[tid]) != 0:
                    continue                                 # already routed (sanitize/static/added)
                tids.append(int(tid))
                inits.append(base.weight[tid].detach().float())
                row_map.append((None, int(tid)))
                caption_tids.append(int(tid))
            if caption_tids:
                print(f"[concord] caption-vocab: TE{te_idx} +{len(caption_tids)} base-vocab token(s) "
                      f"now trainable (seeded from base.weight)")
        # anchor is per-CORE: a caption-ONLY plane honors concord_caption_vocab_anchor (default False, so
        # the emb servo's coherence climb engages); any added token keeps concord_embedding_anchor.
        _has_added = any(e is not None for (e, _k) in row_map)
        anchor_flag = (bool(getattr(config, "concord_caption_vocab_anchor", False))
                       if (caption_tids and not _has_added)
                       else bool(getattr(config, "concord_embedding_anchor", True)))
        if tids:
            inits_t = torch.stack(inits).to(base.weight.device)
            # (a) PRESERVE each token's own norm instead of pinning to the vocab median. The
            # median pin homogenizes base-vocab tokens -- cutting high-norm glue (',', 'the')
            # down to the median and boosting rare tokens up -- a norm-identity rewrite that
            # reads as huge spurious "movement" with ~zero direction change. Per-row target =
            # the seeded init norm keeps base-vocab tokens at their real norm. Default on;
            # set concord_embedding_preserve_norm=False to restore median pinning. NOTE: full
            # effect on a FRESH run -- on resume the init re-seeds from the already-materialized
            # (previously pinned) embedding, so the norm is only un-flattened from a clean start.
            _preserve = bool(getattr(config, "concord_embedding_preserve_norm", True))
            _tgt = inits_t.float().norm(dim=1) if _preserve else median
            cp.attach_trainable(tids, inits_t, lr, _tgt, anchor=anchor_flag)
            if q_on:
                # Tag matching, TWO routes: (1) added-embedding rows by placeholder name (the
                # original path); (2) caption-vocab rows by TOKEN DECOMPOSITION -- each tag WORD
                # is tokenized and ALL its BPE fragment ids claim their rows ('uscld' -> usc+ld,
                # 'stlzd' -> stl+z+d, single-token words -> themselves). Without (2) a caption-
                # vocab-only plane matches NOTHING (rows are (None, tid); placeholders don't
                # exist) and the shield silently skips -- the quality axes then sit unabsorbed
                # in the accumulator (measured: 4 near-single-owner components, 66-69% of
                # accumulated energy, exactly the tag fragments).
                tag_tids = set()
                for _w in q_tag_set:
                    try:
                        tag_tids.update(int(t) for t in
                                        tokenizer(_w, add_special_tokens=False).input_ids)
                    except Exception:
                        pass
                is_tag = torch.tensor([(emb is not None and emb.placeholder in q_tag_set)
                                       or (emb is None and int(_k) in tag_tids)
                                       for (emb, _k) in row_map], dtype=torch.bool)
                if bool(is_tag.any()) and bool((~is_tag).any()):
                    tag_idx = torch.nonzero(is_tag, as_tuple=False).reshape(-1)
                    cp.trainable.set_quality_shield(tag_idx, (~is_tag).float(),
                                                    base.weight.float().mean(0), q_one_sided)
                    cp.trainable.update_quality_basis()           # seed Q from the init vectors
                    print(f"[concord] quality-tag shield: TE{te_idx} {int(is_tag.sum())} tag row(s), "
                          f"{int((~is_tag).sum())} subject row(s) shielded "
                          f"({'one-sided (block toward bad)' if q_one_sided else 'hard (orthogonal)'})")
                elif bool(is_tag.all()):
                    print(f"[concord] quality-tag shield: TE{te_idx} ALL rows are tags -> nothing to "
                          f"shield (need a non-tag subject embedding); skipped")
                else:
                    print(f"[concord] quality-tag shield: TE{te_idx} no rows matched tag list "
                          f"{sorted(q_tag_set)}; skipped")
                if s_tag_set:
                    # STYLE tags: same two matching routes as quality
                    # (placeholder name; BPE fragments for caption-vocab
                    # rows). Two-sided by design -- see set_style_shield.
                    stag_tids = set()
                    for _w in s_tag_set:
                        try:
                            stag_tids.update(int(t) for t in
                                             tokenizer(_w, add_special_tokens=False).input_ids)
                        except Exception:
                            pass
                    is_stag = torch.tensor([(emb is not None and emb.placeholder in s_tag_set)
                                            or (emb is None and int(_k) in stag_tids)
                                            for (emb, _k) in row_map], dtype=torch.bool)
                    if bool(is_stag.any()) and bool((~is_stag).any()):
                        style_idx = torch.nonzero(is_stag, as_tuple=False).reshape(-1)
                        cp.trainable.set_style_shield(style_idx, (~is_stag).float(),
                                                      base.weight.float().mean(0))
                        cp.trainable.update_quality_basis()   # seed BOTH bases
                        print(f"[concord] style-tag shield: TE{te_idx} {int(is_stag.sum())} style "
                              f"row(s) free as the style sink; {int((~is_stag).sum())} row(s) "
                              f"hard-projected (two-sided) off their span")
                    else:
                        print(f"[concord] style-tag shield: TE{te_idx} no usable style/subject "
                              f"split from {sorted(s_tag_set)}; skipped")
            # Group-subspace shaping (exp61): drive off each embedding's `group` label. Own
            # toggles, independent of the quality-orthogonal shield above. Caption-vocab rows
            # (emb is None) and blank-group rows are ungrouped (-1).
            _g_sep = bool(getattr(config, "concord_emb_group_separate", False))
            _g_flat = bool(getattr(config, "concord_emb_group_flatten", False))
            if tids and (_g_sep or _g_flat):
                _sep_g = float(getattr(config, "concord_emb_group_separate_gamma", 0.5) or 0.5)
                _flat_g = float(getattr(config, "concord_emb_group_flatten_gamma", 0.5) or 0.5)
                # The `group` label lives on the TrainEmbeddingConfig, NOT on the model's
                # BaseModelEmbedding (emb in row_map). Link by uuid -- the same id used for
                # train_uuids above. Caption-vocab rows (emb is None) are ungrouped.
                _ecs = list(getattr(config, "additional_embeddings", []) or [])
                if getattr(config, "embedding", None) is not None:
                    _ecs.append(config.embedding)
                _uuid2grp = {ec.uuid: (getattr(ec, "group", "") or "").strip()
                             for ec in _ecs
                             if getattr(ec, "uuid", None) and (getattr(ec, "group", "") or "").strip()}
                _gname = [(_uuid2grp.get(getattr(emb, "uuid", None), "") if emb is not None else "")
                          for (emb, _k) in row_map]
                _gnames = sorted({g for g in _gname if g})
                _n2i = {g: i for i, g in enumerate(_gnames)}
                _gids = torch.tensor([_n2i.get(g, -1) for g in _gname], dtype=torch.long)
                # embedding id per row for BLOCK-AWARE flatten: same emb object -> same id, so a
                # multi-token concept's own tokens stay coherent (flattened BETWEEN concepts, not
                # within one). Caption rows (emb None) are each a singleton.
                _e2i, _eid, _eid_list = {}, 0, []
                for (emb, _k) in row_map:
                    if emb is None:
                        _eid_list.append(_eid); _eid += 1
                    else:
                        _key = id(emb)
                        if _key not in _e2i:
                            _e2i[_key] = _eid; _eid += 1
                        _eid_list.append(_e2i[_key])
                _eids = torch.tensor(_eid_list, dtype=torch.long)
                _n_grp = int((_gids >= 0).sum())
                if _n_grp >= 2:
                    # Supervised + config-derived -> re-arms DETERMINISTICALLY on every setup,
                    # INCLUDING RESUME: setup_packed_embeddings re-runs after _setup_embeddings
                    # restores the trained vectors, so no sidecar is needed (unlike the DISCOVERED
                    # common-mode deflate). Survives the sample deactivate/reactivate too (the
                    # trainable object -- and its _group_ids -- persists across the pointer swap).
                    cp.trainable.set_group_shield(_gids, _g_sep, _sep_g, _g_flat, _flat_g, _eids)
                    from collections import Counter as _Counter
                    _members = dict(_Counter(g for g in _gname if g))
                    _concepts = {}
                    for _nm, _ev in zip(_gname, _eid_list):
                        if _nm:
                            _concepts.setdefault(_nm, set()).add(_ev)
                    print(f"[concord] group-subspace INIT: TE{te_idx} armed {len(_gnames)} group(s) "
                          f"over {_n_grp} row(s) ({len(_gname) - _n_grp} ungrouped) | "
                          f"separate={('g=%.2f' % _sep_g) if _g_sep else 'off'} "
                          f"flatten={('g=%.2f' % _flat_g) if _g_flat else 'off'} | eager/bridge-TE path",
                          flush=True)
                    for _gn in _gnames:
                        print(f"[concord] group-subspace INIT:   TE{te_idx} group '{_gn}': "
                              f"{_members[_gn]} row(s) across {len(_concepts.get(_gn, ()))} concept(s)",
                              flush=True)
                else:
                    print(f"[concord] group-subspace INIT: TE{te_idx} SKIPPED -- need >=2 grouped rows "
                          f"(found groups={_gnames}, {_n_grp} grouped row(s))", flush=True)
        te.text_model.embeddings.token_embedding = cp
        planes.append({"te_idx": te_idx, "te": te, "cp": cp, "base": base, "row_map": row_map,
                       "caption_tids": caption_tids})
    model.concord_control_planes = planes
    # bring the packed cores under the controller's physics (friction at the
    # embedding lr, warmup+cosine schedule, deploy-bridge inclusion, divot delay)
    _ctrl = getattr(model, "concord_controller", None)
    if _ctrl is not None:
        _ws = getattr(config, "workspace_dir", None)
        _ctrl.register_embedding_cores(
            planes, lr,
            delay_epochs=float(getattr(config, "concord_embedding_delay_epochs", 0.0) or 0.0),
            auto_drive=bool(getattr(config, "concord_embedding_auto_drive", False)),
            freq_exponent=float(getattr(config, "concord_embedding_freq_exponent", 0.5)),
            calib_path=(str(Path(_ws) / "concord_embedding_calibration.json") if _ws else None),
            window_report=bool(getattr(config, "concord_embedding_window_report", False)),
            deflate=bool(getattr(config, "concord_emb_deflate", False)),
            deflate_gamma=float(getattr(config, "concord_emb_deflate_gamma", 0.5) or 0.5),
            deflate_modes=int(getattr(config, "concord_emb_deflate_modes", 8) or 8),
            noise_seed=bool(getattr(config, "concord_emb_noise_seed", False)))
    # The plain-SGD wrapper path is bypassed; UNHOOK it before dropping the refs. The wrapper
    # monkeypatches the ORIGINAL token_embedding's forward, which is now the control plane's
    # .base -- so a base-route token (cp.base(flat)) would still run AdditionalEmbeddingWrapper
    # .forward, which cats in the scattered additional-embedding .vectors. Those .vectors aren't
    # in the TE module tree, so they don't follow text_encoder_*_to(device) -> the graph's
    # encode_text (which runs the full TE forward, both encoders) crashes with a CPU-vs-CUDA
    # mismatch. Unhooking restores cp.base to a plain nn.Embedding (the additional tokens are
    # served by cp.trainable, not the wrapper). Then drop the refs so after_optimizer_step's
    # preserve_embedding_norm guard short-circuits.
    for _w in (getattr(model, "embedding_wrapper_1", None), getattr(model, "embedding_wrapper_2", None)):
        if _w is not None:
            _w.remove_hook_from_module()
    model.embedding_wrapper_1 = None
    model.embedding_wrapper_2 = None
    rows = len(planes[0]["row_map"]) if planes else 0
    anchored = bool(getattr(config, "concord_embedding_anchor", True))
    auto = bool(getattr(config, "concord_embedding_auto_drive", False))
    delay = float(getattr(config, "concord_embedding_delay_epochs", 0.0) or 0.0)
    extra = ""
    if delay > 0:
        beta = float(getattr(config, "concord_embedding_freq_exponent", 0.5))
        extra = (f";  divot: frozen {delay:g} epoch(s)"
                 + (f", auto-drive calibration armed (beta={beta:g})" if auto else ""))
    elif auto:
        extra = ";  auto-drive requested but delay=0 -> NO calibration window (uniform drive)"
    _preserve = bool(getattr(config, "concord_embedding_preserve_norm", True))
    _norm = ("ANCHORED (init frozen in v_slow, deploy = init + gated delta)" if anchored
             else "deploy-norm preserved at each token's own base norm" if _preserve
             else "deploy-norm pinned to vocab median")
    print(f"[concord] packed embeddings ON: {rows} trainable token row(s)/TE, lr={lr}; "
          f"{_norm}; plain-SGD embedding path bypassed{extra}")


def reenable_packed_embedding_grad(model):
    """Keep each trainable's dummy _grad_anchor requires_grad=True so its self-step autograd
    Function fires. The TE freeze in _setup_model_part_requires_grad (text_encoder_* frozen
    for embedding-only training) turns it off; this restores it. Called at setup AND every
    after_optimizer_step (both run __setup_requires_grad)."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        cp = plane["cp"]
        if cp.trainable is not None:
            cp.trainable._grad_anchor.requires_grad_(True)


def materialize_packed_embeddings_to_vectors(model):
    """Copy each trained token's DEPLOY vector (s_slow+v_slow, dropping the noisy s_fast)
    into the matching embedding.*.vector, so OneTrainer's embedding saver writes them as the
    standard clip_l/clip_g safetensors (portable). Call before any save."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        cp = plane["cp"]
        if cp.trainable is None:
            continue
        deploy = cp.trainable.deploy_weight().detach()           # [K, dim]
        # Quality-tag shield, HARD guarantee on the SAVED artifact: project the
        # SUBJECT rows off the final tag subspace before they become the portable
        # safetensors; tag rows save AS LEARNED (they are the quality knob). The
        # gradient shield already kept subjects clean along the trajectory, so this
        # is near-no-op; the printed residual is what the dynamics left (~0).
        q_Q = getattr(cp.trainable, "_quality_Q", None)
        if q_Q is not None and q_Q.numel() > 0:
            # An OPTIONAL polish must never block a checkpoint: if anything here
            # fails, warn and save the unprojected deploy (the per-step gradient
            # shield already kept subjects clean during training).
            try:
                from quality_orthogonal import QualityProjector
                dev = deploy.device                              # saver may materialize on CPU
                proj = QualityProjector(q_Q.to(dev), cp.trainable._quality_mu.to(dev))
                mask = cp.trainable._quality_subject_mask.to(dev).bool()   # [K,1]
                sub = mask.reshape(-1)
                one_sided = getattr(cp.trainable, "_quality_one_sided", False)
                nrm = deploy[sub].float().norm(dim=1).median().clamp_min(1e-12).item()
                tb = proj.toward_overlap(deploy[sub])            # the bad lean we remove
                cleaned = proj.project_positions(deploy, one_sided)
                deploy = torch.where(mask, cleaned, deploy)      # subjects cleaned, tags as-is
                ta = proj.toward_overlap(deploy[sub])
                # one-sided keeps the away-from-bad (good) component, so |overlap|
                # stays nonzero BY DESIGN -- report it separately, not as the result.
                extra = (f"; away-from-bad kept (|overlap| {proj.overlap(deploy[sub]):.2e})"
                         if one_sided else "")
                print(f"[concord] quality-tag shield: TE{plane['te_idx']} subjects cleaned for save "
                      f"({'one-sided' if one_sided else 'hard'}): toward-bad overlap "
                      f"{tb:.2e} -> {ta:.2e} ({100 * tb / nrm:.1f}% -> {100 * ta / nrm:.1f}% of row norm)"
                      f"{extra}")
            except Exception as e:
                print(f"[concord] quality-tag shield: save projection skipped for TE{plane['te_idx']} "
                      f"({type(e).__name__}: {e}); saving unprojected deploy "
                      f"(training-time gradient shield still applied)")
        with torch.no_grad():
            base = plane["base"]
            for row, (emb, k) in enumerate(plane["row_map"]):
                if emb is None:                              # caption-vocab BASE token: no .vector;
                    base.weight[k].copy_(deploy[row].to(  # k is the tid -> write deploy to base.weight[tid]
                        dtype=base.weight.dtype, device=base.weight.device))
                else:
                    emb.vector[k].copy_(deploy[row].to(dtype=emb.vector.dtype, device=emb.vector.device))


def deactivate_packed_embeddings(model):
    """Temporarily restore each TE's ORIGINAL token_embedding so it serializes as a standard
    CLIPTextModel (the control-plane buffers must not enter its state_dict). Pair with
    reactivate_packed_embeddings in a try/finally around the save."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        plane["te"].text_model.embeddings.token_embedding = plane["base"]


def reactivate_packed_embeddings(model):
    """Re-install the control planes after a save (inverse of deactivate_packed_embeddings)."""
    planes = getattr(model, "concord_control_planes", None)
    if not planes:
        return
    for plane in planes:
        plane["te"].text_model.embeddings.token_embedding = plane["cp"]
