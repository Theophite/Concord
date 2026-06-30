"""Canonical Concord config projections (consolidation step 1 — additive, no behavior change).

Today FOUR config layers carry the same knobs and can disagree (REFACTOR_PLAN §6):
  WINNER dict        (concord_winner.py)  -- the swap recipe
  ConcordConfig      (concord_winner.py)  -- the dataclass schema  ← CANONICAL source
  CONCORD_DEFAULTS   (optimizer_util.py OPTIMIZER_DEFAULT_PARAMETERS[CONCORD]) -- the GUI/config-file panel
  getattr fallbacks  (concord_ot.py)      -- TrainConfig Tier-B knobs

This module makes `ConcordConfig` the single source: WINNER and CONCORD_DEFAULTS are
PROJECTIONS of a `ConcordConfig()` instance, modulo a small set of INTENTIONAL
per-surface overrides (below). `tests/test_config_projection.py` asserts the current
literals EQUAL these projections, so future drift is caught at atol=0.

Scope of THIS step: additive only. The dataclass still lives in `concord_winner`; the
projection logic lives here. A later step moves `ConcordConfig` into this file and has
`concord_winner` re-export it (O5 needs `from concord_winner import ConcordConfig` to keep
working). CONCORD_DEFAULTS stays a spelled-out AST-evaluable LITERAL in optimizer_util
(test_doc_config AST-parses it with only {'Optimizer': Optimizer} in scope) — the literal
remains the doc-test's source; THIS projection is the equality backstop (O5 / Hole 6).
"""

# ── WINNER: the swap recipe. PURE projection of ConcordConfig (no overrides). ──────
WINNER_KEYS = (
    "alpha", "alpha_v_fast", "weight_decay", "eps",
    "step_cap", "v_scale", "precond_p", "gf_trust_delta_sq",
    "gf_consol",
    "ratio_chase_floor", "ratio_chase_floor_min",
    "ratio_leak_floor", "ratio_leak_floor_min",
    "sigmag_iso", "sigmag_peak",
    "lr_min_frac",
)

# ── CONCORD_DEFAULTS: the GUI panel. Projection of ConcordConfig for the shared keys,
#    PLUS the panel-specific overrides where the LIVE default deliberately differs from
#    the bare-dataclass (test-only) picker default. ────────────────────────────────
PANEL_KEYS = (
    "weight_decay", "noise", "sigmag_peak", "lazy_gate", "lazy_active_thresh",
    "warmup", "lr_min_frac", "step_cap", "gf_trust_delta_sq", "min_leak",
    "evap_build_min", "lamb_trust", "lamb_cap", "lamb_clip", "beta2",
    "beta2_epoch_window", "vhat_warmstart", "bias_correct_v", "coh_vhat",
    "coh_kappa", "dissipation_fill_ramp", "telescope_epoch_window",
    "dissipation", "autotune_table", "autotune_reprobe_band",
    "autotune_gamma_snr_on", "autotune_gamma_snr", "autotune_beta1_on",
    "autotune_beta1_coh", "autotune_servo", "autotune_climb_rate",
    "autotune_boil_ceiling",
)

# INTENTIONAL per-surface overrides (the four real differences, NOT garbage):
#   - momentum: the AUX-SGD momentum (norms/biases); NOT a ConcordConfig field -> panel-only.
#   - dissipation/autotune_table/autotune_reprobe_band: the LIVE shipped recipe ENABLES
#     dimensionless dissipation (lam=0.025) + the CPU-calibrated autotune table + the exp-11d
#     reprobe watchdog by default; the bare ConcordConfig() picker leaves them OFF (None). See O7.
PANEL_OVERRIDES = {
    "momentum": 0.9,
    "dissipation": 0.025,
    "autotune_table": "[[0.387,0],[0.314,0.1],[0.288,0.2],[0.274,0.4],[0.256,0.4]]",
    "autotune_reprobe_band": 0.02,
}


def winner_from(cfg):
    """Project a ConcordConfig instance to the WINNER swap-recipe dict."""
    return {k: getattr(cfg, k) for k in WINNER_KEYS}


def concord_defaults_from(cfg):
    """Project a ConcordConfig instance to the CONCORD_DEFAULTS panel dict:
    the shared keys read from cfg, then the intentional panel overrides applied."""
    d = {k: getattr(cfg, k) for k in PANEL_KEYS}
    d.update(PANEL_OVERRIDES)
    return d
