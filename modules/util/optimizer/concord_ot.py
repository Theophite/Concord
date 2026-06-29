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
        autotune_servo=bool(pick("autotune_servo", d.autotune_servo)),
        autotune_climb_rate=float(pick("autotune_climb_rate", d.autotune_climb_rate)),
        autotune_boil_ceiling=float(pick("autotune_boil_ceiling", d.autotune_boil_ceiling)),
        dissipation=pick("dissipation", d.dissipation),
    )


class ConcordController:
    """Holds the swapped Concord UNet layers + the per-step schedule + the rebalance gate
    for one training run. Created in the SDXL setup (after the model is loaded, before the
    optimizer is built); driven by the trainer via before_step()/after_step()."""

    def __init__(self, unet, device, learning_rate: float, total_steps: int, optimizer_config=None,
                 module_filters=None, text_encoder=None, te_lr=None, te_wd_anchor=0.5,
                 text_encoder_2=None, te2_lr=None, te_chase_alpha=None,
                 te_use_anchor=True, te2_use_anchor=True):
        from concord_winner import swap_unet_to_winner, GatedRebalance, swap_text_encoder_to_anchor, \
            swap_text_encoder_to_winner, \
            set_lazy_gate, set_lazy_thresh, set_min_leak, set_evap_build_min, set_lamb_trust, \
            set_coh_vhat, set_coh_kappa, set_evap_slack
        self.config = make_concord_config(learning_rate, optimizer_config)
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
        self.layers = swap_unet_to_winner(
            unet, device, self.config.lr, gf_consol=self.config.gf_consol,
            step_cap=self.config.step_cap, gf_trust_delta_sq=self.config.gf_trust_delta_sq,
            verbose=False, module_filters=module_filters,
            train_cond_embed=bool(getattr(self.config, "concord_train_cond_embed", False)),
            conv_full_vhat=bool(getattr(self.config, "concord_conv_full_vhat", False)))
        self.gate = GatedRebalance(self.layers)
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
        self.autotuner = None
        self.autotuners = []   # per-group servos (unet + winner TEs); self.autotuner aliases the unet one
        self._autotune_pending = bool(getattr(self.config, "autotune_servo", False)) \
            or bool(getattr(self.config, "autotune_table", None))
        self._snr_mod_announced = False
        self._current_fill_ramp = 1.0
        # Live [loss]-line dissipation telemetry while the servo owns the boil/memgap meters: the
        # readers PEEK the per-layer buffers non-destructively (the servo keeps the per-epoch reset)
        # and return the per-call DELTA, replacing the stale per-epoch _agg cache. Each reader tracks
        # its own cumulative baseline + the servo epoch it last saw; an _epoch change means the servo
        # just zeroed the meters at a boundary -> rebaseline so it isn't read as a negative spike.
        self._gap_peek_cum = 0.0
        self._gap_peek_epoch = -1
        self._boil_peek_cum = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._boil_peek_epoch = -1
        self._last_boil_protected = None   # cf-discounted boil, set by read_flow_audit for the TB log
        self._last_m6a = None   # M6a dissipation-space diversity meter ([4]/[5]), set by read_flow_audit (log-only)
        # Per-layer servo kappa restored from the backup sidecar (concord_servo.json), consumed when
        # _build_autotuner constructs the EpochDissipationServo so the climb survives the per-epoch
        # exit-42 relaunch instead of resetting to the scalar seed.
        self._servo_resume_state = None
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
        from prototype_packed_b import DissipationAutoTuner, EpochDissipationServo
        self._autotune_pending = False
        if bool(getattr(self.config, "autotune_servo", False)):
            # Per-layer, table-free epoch servo (the redesign). gf_consol is the
            # seed kappa (= dissipation/lr). The kernel's consolidation branch is
            # baked IN at capture only if gf_consol > 0, so require it (same rule
            # as the table tuner). The servo then climbs each layer's kappa
            # one-sided from this seed; no probe window, no table.
            if self.config.gf_consol <= 0:
                print("[concord] autotune_servo ON but gf_consol<=0 -> the kernel "
                      "consolidation branch bakes OUT at capture; servo DISABLED. "
                      "Set dissipation (or gf_consol) > 0 to seed it.")
                self.autotuner = None
                return
            _real_spe = int(self.steps_per_epoch) if self.steps_per_epoch > 0 \
                else max(1, self.total_steps // 100)
            # Servo cadence: actuate N times per epoch (default 3, was 1) by shrinking the
            # servo's window to steps_per_epoch / N. The firing (t % epoch_steps == 0), the
            # boil/memgap meter windows (read+zeroed each firing) and the half-window baseline
            # all retime together -> N x more responsive; no other servo logic is touched.
            # N=1 restores the legacy once-per-epoch cadence. Override: config.autotune_servo_per_epoch.
            _servo_per_epoch = max(1, int(getattr(self.config, "autotune_servo_per_epoch", 3)))
            epoch_steps = max(1, _real_spe // _servo_per_epoch)
            self.autotuners = []
            def _mk_servo(_lyr, _lr, _seed, _nm):
                _s = EpochDissipationServo(
                    _lyr, lr=_lr, seed_kappa=_seed, epoch_steps=epoch_steps,
                    climb_rate=float(getattr(self.config, "autotune_climb_rate", 0.5)),
                    boil_ceiling=(float(getattr(self.config, "concord_servo_protected_boil_ceiling", 0.50))
                                  if bool(getattr(self.config, "concord_servo_protected_boil", True))
                                  else float(getattr(self.config, "autotune_boil_ceiling", 0.05))),
                    beta1_on=self.config.autotune_beta1_on,
                    beta1_coh_floor=self.config.autotune_beta1_coh,
                    protected_boil=bool(getattr(self.config, "concord_servo_protected_boil", True)),
                    waste_ceiling=float(getattr(self.config, "concord_servo_waste_ceiling", 0.12)),
                    name=_nm)
                self.autotuners.append((_nm, _s))
                return _s
            # UNet servo; self.autotuner aliases it for the gamma-SNR hook + the [loss]-line readers.
            self.autotuner = _mk_servo(self.layers, self.config.lr, self.config.gf_consol, "unet")
            # Per-group winner-TE servos: each SINGLE-LR so cap/floor/seed are correct, seeded at the
            # SAME dimensionless lam as the UNet (lam/te_lr, NOT the UNet-lr gf_consol, which under-seeds
            # them ~10x). Each servo's __init__ registers per-layer meters, overriding the
            # _te_boil_scratch sink so it reads its own layers.
            _lam = self.config.gf_consol * self.config.lr          # = dissipation (lr-invariant)
            for _gi, (_te_lyr, _te_lr) in enumerate(self.te_groups):
                _winner = [m for m in _te_lyr if getattr(m, "alpha_v_fast", 0.0) > 0.0]
                if _winner and _te_lr > 0:
                    _mk_servo(_winner, _te_lr, _lam / max(_te_lr, 1e-12), f"te{_gi + 1}")
            # 4th group: the NON-anchored packed token-embedding cores (caption-vocab, and any
            # non-anchored added tokens), at the embedding lr. Gate on alpha_v_fast>0, mirroring the
            # te_groups winner filter: an ANCHORED core (added-token default) has drift_cancel_C=0, so
            # its boil meter is structurally 0 (write_boil False) while memgap still writes -- it would
            # bypass the boil-ceiling brake and ratchet kappa on memgap alone, over-dissipating the
            # anchored delta. Leave anchored cores on register_embedding_cores' fixed kappa_emb. The
            # sighting gate gives per-row sparsity, so a SCALAR kappa per table is correct.
            _emb_live = [m for m in (getattr(self, "emb_cores", None) or [])
                         if getattr(m, "alpha_v_fast", 0.0) > 0.0]
            if _emb_live and getattr(self, "emb_lr", 0.0) > 0:
                _mk_servo(_emb_live, self.emb_lr, _lam / max(self.emb_lr, 1e-12), "emb")
            # Restore each group from the sidecar ({group: state}; a legacy single-servo dict -> unet).
            if self._servo_resume_state is not None:
                _st = self._servo_resume_state
                if isinstance(_st, dict) and "kappa" in _st:       # legacy single-servo sidecar
                    _st = {"unet": _st}
                for _nm, _sv in self.autotuners:
                    _slice = _st.get(_nm) if isinstance(_st, dict) else None
                    if _slice is None:
                        continue
                    try:
                        if _sv.import_state(_slice):
                            _ks = sorted(_sv._kappa.values())
                            print(f"[concord] servo[{_nm}] RESTORED from sidecar: {len(_sv.layers)} "
                                  f"layers, epoch {_sv._epoch}, kappa min/med/max="
                                  f"{_ks[0]:.0f}/{_ks[len(_ks) // 2]:.0f}/{_ks[-1]:.0f}", flush=True)
                        else:
                            print(f"[concord] servo[{_nm}] sidecar mismatch -> fresh seed", flush=True)
                    except Exception as _e:
                        print(f"[concord] servo[{_nm}] sidecar restore failed ({_e}) -> fresh seed",
                              flush=True)
                self._servo_resume_state = None
            return
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

    def register_embedding_cores(self, planes, emb_lr, delay_epochs=0.0,
                                 auto_drive=False, freq_exponent=0.5,
                                 calib_path=None, window_report=False):
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
        self.emb_cores = [t.core for t in self.emb_trainables]
        self.emb_row_names = next((
            [(emb.placeholder if emb is not None else f"base:{_k}") for emb, _k in p["row_map"]]
            for p in planes if p.get("row_map")), [])
        self.emb_lr = float(emb_lr)
        self.emb_delay_epochs = max(0.0, float(delay_epochs))
        self.emb_auto_drive = bool(auto_drive)
        self.emb_freq_exponent = max(0.0, float(freq_exponent))
        self.emb_calib_path = calib_path
        self.emb_window_report = bool(window_report)
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
        if not self.config.telescope_epoch_window or steps_per_epoch <= 0:
            return
        new_av = 1.0 / (2.0 * float(steps_per_epoch))
        old_av = self.config.alpha_v_fast
        self.config.alpha_v_fast = new_av
        for m in self.layers:
            m.alpha_v_fast = new_av
            m.drift_cancel_C = compute_drift_cancel_C(
                m.alpha, new_av, mass_preserve=bool(getattr(m, "mass_preserve_v", True)))
        # Winner-recipe TEs telescope too (alpha_v_fast>0); frozen-anchor TEs (==0) stay pinned.
        for m in self.te_layers:
            if getattr(m, "alpha_v_fast", 0.0) > 0.0:
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
        if self.autotuner is not None and getattr(self.autotuner, "per_layer", False):
            # Servo routes meters per-layer (the shared buffer is empty under it). PEEK the per-layer
            # boil meters live and return the per-call delta -- live boil/waste, not the per-epoch
            # _agg cache (which is 0 until the first post-enable boundary, then stale all epoch).
            bsum = torch.stack([m._boil_meter for m in self.layers]).sum(0).tolist()
            cur = (float(bsum[0]), float(bsum[1]), float(bsum[2]),
                   float(bsum[3]) if len(bsum) > 3 else 0.0,
                   float(bsum[4]) if len(bsum) > 4 else 0.0,
                   float(bsum[5]) if len(bsum) > 5 else 0.0)
            ep = int(getattr(self.autotuner, "_epoch", 0))
            if ep != self._boil_peek_epoch:        # servo zeroed the meters at the boundary
                self._boil_peek_epoch = ep
                self._boil_peek_cum = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            pa, pb, pc, pd, pe, pf = self._boil_peek_cum
            a, b, c, d = cur[0] - pa, cur[1] - pb, cur[2] - pc, cur[3] - pd
            e, f = cur[4] - pe, cur[5] - pf   # M6a: e = sum killed^2*coh_raw*infancy-band, f = sum s_slow^2
            self._boil_peek_cum = cur
        else:
            a, b, c = read_boil(self.layers[0].packed_w.device)
            d = e = f = 0.0
        boil = (a / b) if b > 0 else None
        waste = (b / (b + c)) if (b + c) > 0 else None
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
        if self.autotuner is not None and getattr(self.autotuner, "per_layer", False):
            # Servo routes memgap per-layer (the shared buffer is empty under it). PEEK the per-layer
            # meters live and return the per-call delta -- live gap, not the per-epoch _agg cache.
            cur = float(torch.stack([m._memgap_meter for m in self.layers]).sum().item())
            ep = int(getattr(self.autotuner, "_epoch", 0))
            if ep != self._gap_peek_epoch:         # servo zeroed the meters at the boundary
                # Cold start (sentinel epoch -1): the per-layer meter holds the un-zeroed graph-warmup +
                # first-accumulation lump, NOT a per-epoch delta -- absorb it into the baseline (d=0) so
                # the deploy-loss EMA never seeds on it (else deploy_smooth shows ~2e12 then decays). A
                # NORMAL boundary already zeroed the meters, so cur is a fresh delta -> baseline 0 as before.
                self._gap_peek_cum = cur if self._gap_peek_epoch < 0 else 0.0
                self._gap_peek_epoch = ep
            d = cur - self._gap_peek_cum
            self._gap_peek_cum = cur
            return -d
        return -read_memgap(self.layers[0].packed_w.device)

    def export_servo_state(self):
        """Per-GROUP per-layer servo kappa for the backup sidecar, as {group_name: state}; None when
        no per-layer servo is active. (The trainer writes it next to concord_clock.json so the climb
        survives the relaunch; import in _build_autotuner restores each group, legacy dict -> unet.)"""
        out = {}
        for nm, sv in getattr(self, "autotuners", []):
            if getattr(sv, "per_layer", False) and hasattr(sv, "export_state"):
                out[nm] = sv.export_state()
        return out or None

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
        the backward / graph replay. Frozen-anchor TEs (alpha_v==0) are never
        modulated; winner-recipe TEs (alpha_v>0) are, same as the UNet.
        """
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
            _base_l = float(layer._gf_consol_value) if servo else float(base)
            layer._gf_consol_buf.copy_(
                (mod * _base_l * self._current_fill_ramp).clamp_max(_cap_l))

    @torch.no_grad()
    def before_step(self):
        """BEFORE forward/backward: advance the winner schedule onto the layer device
        tensors (lr / sigma / coherence floors) that the fused backward reads."""
        from concord_winner import winner_step
        if self._autotune_pending:
            self._build_autotuner()
        if self.autotuners:
            for _, _sv in self.autotuners:
                _sv.step(self.step_idx)
        elif self.autotuner is not None:
            self.autotuner.step(self.step_idx)
        # D4: anneal the ratio-coh bootstrap floors over ONE EPOCH (like the telescope), not the
        # whole run. winner_step defaults floor_horizon=total_iters when None, which keeps the
        # chase/leak floors high for most of training; steps_per_epoch (set by apply_epoch_window)
        # is the intended window.
        _fh = self.steps_per_epoch if self.steps_per_epoch > 0 else None
        winner_step(self.step_idx, self.total_steps, self.layers, config=self.config,
                    floor_horizon=_fh)
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
                m._gf_consol_buf.fill_(m._gf_consol_value * self._current_fill_ramp)
            for m in self.te_layers:               # winner TEs only; frozen anchors (alpha_v==0,
                if getattr(m, "alpha_v_fast", 0.0) > 0.0:   # gf_consol=0) have no fill-ramp to apply
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
        for _g in self.te_gates:                       # per-encoder gated rebalance
            _g()
        self.step_idx += 1
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
            # build_ok = the soft drain gate's EXPECTED firing rate, mean(min(|s_fast|/evap_build_min, 1)),
            # NOT a >= threshold count. The kernel gate is stochastic (P(drain) = |s_fast|/evap_build_min);
            # a >= count reads ~0 whenever the chase pins |s_fast| below the scale, hiding whether the
            # drain fires at all. This reports the real duty cycle: 0% = truly off, small = pegged low.
            import prototype_packed_b as _ppb
            build_min = float(_ppb._EVAP_BUILD_MIN)
            dev = layers[0].packed_w.device
            acc = torch.zeros(8, dtype=torch.float64, device=dev)
            ntot = 0
            for m in layers:
                p = m.packed_w
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
                acc[6] += (sf.abs() / (build_min + 1e-30)).clamp(max=1.0).sum(dtype=torch.float64)
                ntot += sf.numel()
            return acc.tolist(), ntot

        def _line(tag, layers):
            if not layers:
                return
            a, ntot = _metrics(layers)
            nd, nsf = a[0] ** 0.5, a[3] ** 0.5
            print(f"[concord-health:{tag}] step={self.step_idx} "
                  f"||deploy||={nd:.1f} ||v||={a[1] ** 0.5:.1f} ||s||={a[2] ** 0.5:.1f} "
                  f"||s_fast||={nsf:.3f} build_ok={100.0 * a[6] / max(1, ntot):.3f}% "
                  f"coh={a[4] / max(a[3], 1e-30):.3f} cos(sf,dep)={a[5] / max(nsf * nd, 1e-30):+.3f} "
                  f"gap={a[7] ** 0.5:.1f}",
                  flush=True)

        _line("unet", self.layers)
        _line("te", self.te_layers)   # frozen TE -> coh~0, gap~||W||; creep TE -> coh live, gap grows from ~0

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
    from modules.util.enum.Optimizer import Optimizer
    return (config.optimizer.optimizer == Optimizer.CONCORD
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
    ids = set()

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
                ids.update(int(t) for t in toks)
    ids -= special
    if vocab_size > 0:
        ids = {t for t in ids if t < vocab_size}
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
            cp.attach_trainable(tids, torch.stack(inits).to(base.weight.device), lr, median,
                                anchor=anchor_flag)
            if q_on:
                is_tag = torch.tensor([(emb is not None and emb.placeholder in q_tag_set)
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
            window_report=bool(getattr(config, "concord_embedding_window_report", False)))
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
    print(f"[concord] packed embeddings ON: {rows} trainable token row(s)/TE, lr={lr}; "
          f"{'ANCHORED (init frozen in v_slow, deploy = init + gated delta)' if anchored else 'deploy-norm pinned to vocab median'}; "
          f"plain-SGD embedding path bypassed{extra}")


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
