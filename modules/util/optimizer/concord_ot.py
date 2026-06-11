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
        min_leak=float(pick("min_leak", d.min_leak)),
        evap_build_min=float(pick("evap_build_min", d.evap_build_min)),
        dissipation_fill_ramp=bool(pick("dissipation_fill_ramp", d.dissipation_fill_ramp)),
        telescope_epoch_window=bool(pick("telescope_epoch_window", d.telescope_epoch_window)),
        autotune_table=pick("autotune_table", d.autotune_table),
        autotune_beta1_on=float(pick("autotune_beta1_on", d.autotune_beta1_on)),
        autotune_beta1_coh=float(pick("autotune_beta1_coh", d.autotune_beta1_coh)),
        autotune_reprobe_band=pick("autotune_reprobe_band", d.autotune_reprobe_band),
        autotune_gamma_snr=pick("autotune_gamma_snr", d.autotune_gamma_snr),
        dissipation=pick("dissipation", d.dissipation),
    )


class ConcordController:
    """Holds the swapped Concord UNet layers + the per-step schedule + the rebalance gate
    for one training run. Created in the SDXL setup (after the model is loaded, before the
    optimizer is built); driven by the trainer via before_step()/after_step()."""

    def __init__(self, unet, device, learning_rate: float, total_steps: int, optimizer_config=None,
                 module_filters=None, text_encoder=None, te_lr=None, te_wd_anchor=0.5):
        from concord_winner import swap_unet_to_winner, GatedRebalance, swap_text_encoder_to_anchor, \
            set_lazy_gate, set_lazy_thresh, set_min_leak, set_evap_build_min
        self.config = make_concord_config(learning_rate, optimizer_config)
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
        self.layers = swap_unet_to_winner(
            unet, device, self.config.lr, gf_consol=self.config.gf_consol,
            step_cap=self.config.step_cap, gf_trust_delta_sq=self.config.gf_trust_delta_sq,
            verbose=False, module_filters=module_filters)
        self.gate = GatedRebalance(self.layers)
        # Frozen-anchor TE training (CLIP-L): swapped AFTER the UNet so the shared global coh
        # flags are already set; driven with its own lr. Empty unless a text_encoder is passed.
        self.te_lr = float(te_lr) if te_lr else self.config.lr
        self.text_encoder = text_encoder          # held for the reversible TE deploy bridge
        self.te_layers = (swap_text_encoder_to_anchor(text_encoder, device, self.te_lr, te_wd_anchor)
                          if text_encoder is not None else [])
        self.te_gate = GatedRebalance(self.te_layers) if self.te_layers else None
        # Lazy-update gate is a module-level global read at every kernel launch; swap_unet_to_winner
        # forces the coherence/noise flags but not this one, so set it explicitly from config here.
        set_lazy_gate(self.config.lazy_gate)
        set_lazy_thresh(self.config.lazy_active_thresh)
        # Servo min-leak floor: module global like the lazy gate (per-run constant,
        # baked at CUDA-graph capture). Guards the lam -> 1 regime from slam-shut.
        set_min_leak(self.config.min_leak)
        # Hypothesis-infancy guard: no dissipation below one deploy tick.
        set_evap_build_min(self.config.evap_build_min)
        # Dissipation autotuner (probe-then-commit), opt-in via optimizer.autotune_table.
        # Built LAZILY on the first before_step(): total_steps here is a placeholder —
        # the trainer finalizes the horizon at train start.
        self.autotuner = None
        self._autotune_pending = bool(getattr(self.config, "autotune_table", None))
        self._snr_mod_announced = False
        self._current_fill_ramp = 1.0
        self.emb_cores = []         # packed-embedding cores (register_embedding_cores)
        self.emb_lr = 0.0
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

    def register_embedding_cores(self, planes, emb_lr):
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
        self.emb_cores = [p["cp"].trainable.core for p in planes
                          if p.get("cp") is not None and p["cp"].trainable is not None]
        self.emb_lr = float(emb_lr)
        if not self.emb_cores:
            return
        lam = (float(self.config.dissipation) if self.config.dissipation is not None
               else float(self.config.gf_consol) * float(self.config.lr))
        kappa_emb = lam / max(self.emb_lr, 1e-12)
        for c in self.emb_cores:
            c.gf_consol = kappa_emb
        print(f"[concord] embedding cores registered: {len(self.emb_cores)} TE plane(s) "
              f"under the controller -- lam={lam:g} @ lr_emb={self.emb_lr:g} -> "
              f"kappa_emb={kappa_emb:.0f}; warmup+cosine schedule, sigma off; "
              f"deploy bridge now masks embedding s_fast during sampling")

    @torch.no_grad()
    def apply_epoch_window(self, steps_per_epoch):
        """Pin the telescope window to the dataset revisit period (exp-20
        freshness law): alpha_v_fast = 1/(2*steps_per_epoch), so the anchor
        integrates exactly one full pass before motion counts as drift --
        every example votes once. C* is a function of alpha_v, so it is
        re-derived per layer; the telescope-clock consumers (fill ramp,
        probe floor, watchdog arm delay) read config.alpha_v_fast and follow
        automatically. UNet layers only (the TE anchor is pinned). Call at
        horizon-finalize time, BEFORE the first training step / capture
        (alpha_v_fast and C* are launch-time scalars baked at capture).
        Idempotent across resumes."""
        from prototype_packed_b import compute_drift_cancel_C
        if not self.config.telescope_epoch_window or steps_per_epoch <= 0:
            return
        new_av = 1.0 / (2.0 * float(steps_per_epoch))
        old_av = self.config.alpha_v_fast
        self.config.alpha_v_fast = new_av
        for m in self.layers:
            m.alpha_v_fast = new_av
            m.drift_cancel_C = compute_drift_cancel_C(
                m.alpha, new_av, mass_preserve=bool(getattr(m, "mass_preserve_v", True)))
        print(f"[concord] telescope epoch window: alpha_v {old_av:g} -> {new_av:g} "
              f"(window = {steps_per_epoch:.0f} steps = 1 epoch; C* re-derived; "
              f"fill ramp / probe floor / watchdog follow)", flush=True)

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
        a, b, c = read_boil(self.layers[0].packed_w.device)
        boil = (a / b) if b > 0 else None
        waste = (b / (b + c)) if (b + c) > 0 else None
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
        return -read_memgap(self.layers[0].packed_w.device)

    @torch.no_grad()
    def on_timesteps(self, timesteps, alphas_cumprod):
        """gamma-SNR dissipation modulation (opt-in via optimizer.autotune_gamma_snr).

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
        the backward / graph replay. TE layers are never modulated.
        """
        knee = self.config.autotune_gamma_snr
        if knee is None:
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
        cap = self._LAM_MOD_CAP / max(self.config.lr, 1e-12)
        # compose with the fill ramp (run-level infancy) -- this hook runs after
        # before_step and would otherwise overwrite the ramped buffer value
        kappa_t = (mod * float(base) * self._current_fill_ramp).clamp_max(cap)
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
        for layer in self.layers:
            layer._gf_consol_buf.copy_(kappa_t)

    @torch.no_grad()
    def before_step(self):
        """BEFORE forward/backward: advance the winner schedule onto the layer device
        tensors (lr / sigma / coherence floors) that the fused backward reads."""
        from concord_winner import winner_step
        if self._autotune_pending:
            self._build_autotuner()
        if self.autotuner is not None:
            self.autotuner.step(self.step_idx)
        winner_step(self.step_idx, self.total_steps, self.layers, config=self.config)
        # Run-level infancy (dissipation_fill_ramp): write kappa_base * ramp into
        # the per-layer device buffers each step (the lr/sigma pattern; the host
        # mirror _gf_consol_value keeps the unmodulated base, so tuner commits and
        # this compose cleanly). Don't boil weight off while the pretrained mass
        # is in transit: friction ~0 early, 63% at the telescope time constant,
        # ~full by 3 tau, then the cosine lr takes it down -- rising-late, like
        # the fluctuation sigma. UNet layers only (the TE anchor path is pinned).
        if self.config.dissipation_fill_ramp:
            self._current_fill_ramp = self._fill_ramp(self.step_idx, self.config.alpha_v_fast)
            for m in self.layers:
                m._gf_consol_buf.fill_(m._gf_consol_value * self._current_fill_ramp)
        if self.te_layers:
            winner_step(self.step_idx, self.total_steps, self.te_layers,
                        peak_lr=self.te_lr, config=self.config)
        if self.emb_cores:
            # embeddings get their OWN schedule group: warmup + cosine at the
            # embedding lr, fluctuation off (see register_embedding_cores)
            winner_step(self.step_idx, self.total_steps, self.emb_cores,
                        peak_lr=self.emb_lr, noise=False, config=self.config)

    @torch.no_grad()
    def after_step(self):
        """AFTER the optimizer update: gated rebalance (skips the no-op launches), tick."""
        self.gate()
        if self.te_gate is not None:
            self.te_gate()
        self.step_idx += 1

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
    @torch.no_grad()
    def materialize_unet_deploy(self):
        """Reversible deploy window, ZERO GPU allocations: D2H-copy each packed
        word to CPU (no device temp), extract/stash the s_fast bits CPU-side,
        mask s_fast out of packed_w IN PLACE (the deploy weight is exactly the
        low 16 bits). Fused mode then dequantizes the deploy weight directly;
        cached mode re-materializes its existing buffer. Restore ORs the bits
        back through ONE reused device scratch — no per-layer allocation churn
        (sampling's VRAM fragmentation is the documented wedge; do not feed it)."""
        import prototype_packed_b as ppb
        stash = []
        # embedding cores included: their forward gathers reconstruct from
        # packed_w directly, so masking alone makes sampling deploy-true
        for m in self.layers + self.emb_cores:
            pk_cpu = m.packed_w.detach().to("cpu")          # D2H, no device temp
            stash.append(((pk_cpu >> 16).to(torch.int16)))  # CPU-side extract
            m.packed_w &= 0xFFFF                            # in place
            if not ppb._FUSED_MATMUL:
                wbuf, _, _ = m._ensure_buffers()
                ppb.materialize_packed_bf16(m.packed_w, m.row_exp, m.col_exp,
                                            out=wbuf,
                                            mantissa_bias=m.MANTISSA_BIAS)
        return stash

    @torch.no_grad()
    def restore_unet_deploy(self, stash):
        import prototype_packed_b as ppb
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
            scratch[:n].copy_(word.reshape(-1))              # one H2D, no alloc
            m.packed_w |= scratch[:n].view_as(m.packed_w)
            if not ppb._FUSED_MATMUL:
                wbuf, _, _ = m._ensure_buffers()
                ppb.materialize_packed_bf16(m.packed_w, m.row_exp, m.col_exp,
                                            out=wbuf,
                                            mantissa_bias=m.MANTISSA_BIAS)


    def materialize_te_deploy(self):
        """REVERSIBLE TE deploy: replace each text-encoder ConcordLinearPackedB with a temp
        nn.Linear holding its get_weight() (keeps s_fast -> the ~16-bit deploy, NOT the
        s_fast-dropping consolidated_weight), so the TE serializes as a standard CLIPTextModel.
        Returns a stash; pass it to restore_te_deploy() in a finally so training continues."""
        import torch.nn as nn
        from prototype_packed_b import ConcordLinearPackedB
        if not self.te_layers or self.text_encoder is None:
            return []
        # ONLY the swapped TE transformer Linears (te_layers). The control-plane embedding
        # core is also a ConcordLinearPackedB living inside the TE module tree, but it has its
        # own save bridge (materialize_packed_embeddings_to_vectors) -> must NOT be clobbered.
        te_set = {id(m) for m in self.te_layers}
        stash = []
        for parent in self.text_encoder.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, ConcordLinearPackedB) and id(child) in te_set:
                    w = child.get_weight()
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
    from modules.util.enum.Optimizer import Optimizer
    return (config.optimizer.optimizer == Optimizer.CONCORD
            and bool(getattr(config, "concord_packed_embeddings", False))
            and config.train_any_embedding())


def _packed_trainable_uuids(config):
    """uuids of the additional embeddings that train via the packed core (train=True,
    non-output)."""
    out = set()
    for ec in config.all_embedding_configs():
        if getattr(ec, "train", False) and not getattr(ec, "is_output_embedding", False):
            out.add(ec.uuid)
    return out


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
    lr = float(config.embedding_learning_rate)
    specs = [
        (1, model.text_encoder_1, model.tokenizer_1, model.all_text_encoder_1_embeddings()),
        (2, model.text_encoder_2, model.tokenizer_2, model.all_text_encoder_2_embeddings()),
    ]
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
        if tids:
            cp.attach_trainable(tids, torch.stack(inits).to(base.weight.device), lr, median,
                                anchor=bool(getattr(config, "concord_embedding_anchor", True)))
        te.text_model.embeddings.token_embedding = cp
        planes.append({"te_idx": te_idx, "te": te, "cp": cp, "base": base, "row_map": row_map})
    model.concord_control_planes = planes
    # bring the packed cores under the controller's physics (friction at the
    # embedding lr, warmup+cosine schedule, deploy-bridge inclusion)
    _ctrl = getattr(model, "concord_controller", None)
    if _ctrl is not None:
        _ctrl.register_embedding_cores(planes, lr)
    # The plain-SGD wrapper path is bypassed; ensure both wrapper refs are None so
    # after_optimizer_step's preserve_embedding_norm guard short-circuits (the model's
    # __init__ leaves embedding_wrapper_2 unset).
    model.embedding_wrapper_1 = None
    model.embedding_wrapper_2 = None
    rows = len(planes[0]["row_map"]) if planes else 0
    anchored = bool(getattr(config, "concord_embedding_anchor", True))
    print(f"[concord] packed embeddings ON: {rows} trainable token row(s)/TE, lr={lr}; "
          f"{'ANCHORED (init frozen in v_slow, deploy = init + gated delta)' if anchored else 'deploy-norm pinned to vocab median'}; "
          f"plain-SGD embedding path bypassed")


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
        with torch.no_grad():
            for row, (emb, k) in enumerate(plane["row_map"]):
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
