"""concord.dual_dissipation_flat_ref - SAME-SCALE ('flat') dual-dissipation reference.

Architect ruling 2026-06-28 (see memory concord-no-x128): the x128 slow/fast scale gap is
an unnecessary int16-legacy artifact. All four packed fields are same-scale block-float
MANTISSAE (CARRY = 1); the weight is their PLAIN SUM. The chase fast->slow becomes an EXACT
same-scale transfer - no 128-quantum, no graduation barrier.

INCOMPLETE PROTOTYPE: this file OMITS the denormal exp-claim + denormal-aware rebalance
(renormalize_flat just SR-halves all four fields). That is WRONG for the canonical design:
denormals are INTRINSIC to rank-1 block-float (a per-row/col exponent can't normalize every
element; an element sub-unit at the shared exponent is denormal and needs the exp-claim for
reach). Removing the ×128 only changes the denormal predicate (coarse==0 -> sub-unit) and
makes denormals RARER, not gone. CIFAR scored 0.595 here only because its weights rarely go
sub-unit. The CANONICAL same-scale design MUST re-add the exp-claim + a denormal-aware
rebalance (HOLE-3 stays). See memory concord-no-x128.

THIS IS THE CANONICAL dual-dissipation (default via dual_dissipation_nn) -- not a variant or
a flag. CIFAR-confirmed: same-scale 0.595 > x128 0.533 @ 6k/40ep (closing half the gap to
AdamW's 0.654), with the fast register staying drained (|e|~0.14 vs x128's ~47 piled sub-grid).
The x128 DualDissipationLayer in dual_dissipation_ref is the DEPRECATED artifact.
It REUSES the validated helpers from dual_dissipation_ref (unpack/pack, coherence, SR, scale,
substrate) and keeps the SAME dynamics (per-arm coherence, pre-evap chase, bracketed evap,
leak->anchor, two-gradient split). Only the SCALE changes (CARRY=1, exact chase); the denormal
exp-claim is a TODO for the canonical design (see the INCOMPLETE PROTOTYPE note above).

Format: int32 word = [e_H | e_L | s_slow | v_slow], each int8, ALL x1 (CARRY = 1).
  live   weight = substrate + (e_H + e_L + s_slow + v_slow) * scale
  deploy weight = substrate + (s_slow + v_slow) * scale   (drops the fast hypotheses -- a
                  same-scale OMISSION, not a grid snap; with the exact chase the fast drains
                  into s_slow so deploy tracks live)

Drop-in for the CIFAR harness: same DualDissipationLayer interface, so
dual_dissipation_nn can be pointed here by swapping DualDissipationLayer.
"""
import math

import torch

from dual_dissipation_ref import (
    unpack_dual, pack_dual, compute_coherence, _sr_round, _hash_uniform,
    _scale_fwd, _pos_hash, make_substrate, INT8_MIN, INT8_MAX, MANTISSA_BIAS,
    SUBSTRATE_XAVIER, SUBSTRATE_KAIMING)

try:
    from dual_dissipation_ref import M_MIN, BRACKET_D
except Exception:                                   # pragma: no cover
    M_MIN, BRACKET_D = 4, 0.5

CARRY_FLAT = 1                                       # THE point: one scale, exact transfers
_RENORM_THRESH = 110                                 # per-row saturation guard (< int8 127)

# SUBSTRATE-DOMINATION TEST (architect, 2026-06-28): when True, load_weights puts the init
# into the TRAINABLE counters (gap-zero s_slow/v_slow, which persist) with substrate=0,
# instead of freezing it in the fp substrate. Tests whether the frozen-random substrate
# component is what caps from-scratch accuracy.
INIT_IN_COUNTERS = False

# v̂-ON-CONVOLUTIONS test (2026-06-28): when True, use a FULL per-element second-moment EMA
# (v_full [N,K]) instead of the rank-1 Adafactor v_row⊗v_col. Measured: the conv 2nd moment is
# ~2× less rank-1 than FC (conv rank1_err ~0.25-0.38 vs fc ~0.13-0.16). Production analog =
# USE_FULL_V / concord_conv_full_vhat (Conv-only, default off).
FULL_VHAT = False


# ============================================================================
# Decode (plain SUM * scale + substrate).  No x128, no denormal render.
# ============================================================================
def _sub_addend(substrate, N, K, device):
    if substrate is None:
        return torch.zeros(N, K, dtype=torch.float32, device=device)
    return substrate.to(torch.float32)


def decode_live_flat(packed, row_exp, col_exp, substrate=None, mantissa_bias=MANTISSA_BIAS):
    e_H, e_L, s_slow, v_slow = unpack_dual(packed)
    N, K = packed.shape
    mant = (e_H + e_L + s_slow + v_slow).to(torch.float32)
    return _sub_addend(substrate, N, K, packed.device) + mant * _scale_fwd(
        row_exp, col_exp, mantissa_bias)


def decode_deploy_flat(packed, row_exp, col_exp, substrate=None, drop_fast=True,
                       mantissa_bias=MANTISSA_BIAS):
    e_H, e_L, s_slow, v_slow = unpack_dual(packed)
    N, K = packed.shape
    mant = (s_slow + v_slow).to(torch.float32) if drop_fast \
        else (e_H + e_L + s_slow + v_slow).to(torch.float32)
    return _sub_addend(substrate, N, K, packed.device) + mant * _scale_fwd(
        row_exp, col_exp, mantissa_bias)


def renormalize_flat(e_H, e_L, s_slow, v_slow, row_exp, pos, salt, thresh=_RENORM_THRESH):
    """Per-row: where any register saturates (row max|.| > thresh), SR-halve ALL FOUR
    registers and bump row_exp by 1. weight = sum*scale is preserved (sum/2 * 2*scale),
    up to the unbiased SR-halving residual (< 1 unit * the new, coarser scale)."""
    rowmax = torch.maximum(torch.maximum(e_H.abs(), e_L.abs()),
                           torch.maximum(s_slow.abs(), v_slow.abs())).amax(dim=1)   # [N]
    hot = rowmax > thresh
    if not bool(hot.any()):
        return e_H, e_L, s_slow, v_slow, row_exp
    hot2 = hot[:, None]

    def halve(t, sa):
        h = t.to(torch.float32) / 2.0
        return torch.where(hot2, _sr_round(h, t, pos, salt ^ sa), t)

    e_H = halve(e_H, 0x1111); e_L = halve(e_L, 0x2222)
    s_slow = halve(s_slow, 0x3333); v_slow = halve(v_slow, 0x4444)
    row_exp = torch.where(hot, row_exp + 1, row_exp)
    return e_H, e_L, s_slow, v_slow, row_exp


# ============================================================================
# The flat layer (matches DualDissipationLayer's interface).
# ============================================================================
class DualDissipationFlatLayer:
    def __init__(self, N, K, enabled=True, grad_accum_M=8, bracket_d=BRACKET_D, seed=0,
                 substrate_seed=None, substrate_mode=SUBSTRATE_XAVIER):
        self.N, self.K = N, K
        self.grad_accum_M = int(grad_accum_M)
        self.enabled = bool(enabled) and (self.grad_accum_M >= M_MIN)
        self.bracket_d = float(bracket_d)
        self.device = torch.device("cpu")
        self.packed_w = torch.zeros(N, K, dtype=torch.int32)
        self.row_exp = torch.zeros(N, dtype=torch.int32)
        self.col_exp = torch.zeros(K, dtype=torch.int32)
        self.v_row = torch.zeros(N, dtype=torch.float32)
        self.v_col = torch.zeros(K, dtype=torch.float32)
        self.v_full = None                               # per-element v̂ (FULL_VHAT), lazily allocated
        self._salt = int(seed) & 0x7FFFFFFF
        self._step = 0
        self.substrate = None
        self.substrate_seed = int(seed if substrate_seed is None else substrate_seed)
        self.substrate_mode = substrate_mode

    @torch.no_grad()
    def load_weights(self, W, base=None):
        """ENABLED: substrate := base (finetune / pre-scaled from-scratch) or a fresh seeded
        draw; row_exp set so one register (~127) spans ~max|substrate| (scale ~ max_abs/128 --
        COARSER than the x128 ref so the int8 sum has range to learn a full-magnitude offset);
        offset zeroed -> weight == substrate."""
        W = W.to(torch.float32)
        self.device = W.device
        dev = self.device
        self.v_row = torch.zeros(self.N, dtype=torch.float32, device=dev)
        self.v_col = torch.zeros(self.K, dtype=torch.float32, device=dev)
        self.v_full = None
        prior = (base.to(torch.float32).to(dev).clone() if base is not None
                 else make_substrate(self.N, self.K, self.substrate_seed,
                                     self.substrate_mode, device=dev))
        max_abs = prior.abs().amax(dim=1).clamp(min=1e-30)
        # scale = 2^(row_exp - bias) ~ max_abs / 128  => v_slow(+-127) spans ~+-max_abs.
        self.row_exp = torch.ceil(torch.log2(max_abs) + 8.0).clamp(-30, 30).to(torch.int32)
        self.col_exp = torch.zeros(self.K, dtype=torch.int32, device=dev)
        z = torch.zeros(self.N, self.K, dtype=torch.int32, device=dev)
        if INIT_IN_COUNTERS:
            # SUBSTRATE-DOMINATION TEST: init -> TRAINABLE counters (gap-zero s_slow/v_slow,
            # which persist), substrate = 0. The deploy weight == the init, but now trainable.
            self.substrate = torch.zeros_like(prior)
            scale = _scale_fwd(self.row_exp, self.col_exp)
            mant = (prior / scale).round().clamp(2 * INT8_MIN, 2 * INT8_MAX)
            v = (mant / 2.0).round().clamp(INT8_MIN, INT8_MAX).to(torch.int32)
            s = (mant - v).clamp(INT8_MIN, INT8_MAX).to(torch.int32)
            self.packed_w = pack_dual(z, z, s, v)
        else:
            self.substrate = prior
            self.packed_w = pack_dual(z, z, z, z)
        self._step = 0

    @torch.no_grad()
    def live_weight(self):
        return decode_live_flat(self.packed_w, self.row_exp, self.col_exp, self.substrate)

    @torch.no_grad()
    def deploy_weight(self, step_salt=0, deterministic=False, drop_fast=True):
        return decode_deploy_flat(self.packed_w, self.row_exp, self.col_exp,
                                  self.substrate, drop_fast=drop_fast)

    @torch.no_grad()
    def step(self, grad_W, lr, *, grad_W_H=None, alpha=0.1, gf_consol=0.0,
             drift_cancel_C=0.02, alpha_v_fast=0.001, coh_kappa=1.0, v_scale=1.0,
             precond_p=0.5, eps=1.0, step_cap=10.0, min_leak=0.0, evap_build_min=128.0,
             beta1=0.0, mantissa_bias=MANTISSA_BIAS, beta2=0.999, use_coh_vhat=True,
             mass_preserve=True, chase_floor=0.05, leak_floor=0.05, consf=1.0, **_ignore):
        self._step += 1
        salt = (self._salt ^ (self._step * 0x9E3779B1)) & 0x7FFFFFFF
        N, K = self.N, self.K
        dev = self.device
        grad_W = grad_W.to(torch.float32).to(dev)
        two_grad = grad_W_H is not None
        if two_grad:
            grad_W_H = grad_W_H.to(torch.float32).to(dev)
        pos = _pos_hash(N, K, device=dev)

        # ---- Adafactor rank-1 vhat (shared preconditioner; mean grad in two-grad) ----
        g_stat = 0.5 * (grad_W + grad_W_H) if two_grad else grad_W
        g2 = g_stat * g_stat
        self.v_row = beta2 * self.v_row + (1 - beta2) * g2.mean(dim=1)
        self.v_col = beta2 * self.v_col + (1 - beta2) * (
            g2.mean(dim=0) / (self.v_row.mean() + 1e-30))
        sum_v_inv = 1.0 / (self.v_row.sum() + 1e-30)
        v_bc = 1.0 / (1.0 - beta2 ** self._step)
        vhat = (self.v_row[:, None] * self.v_col[None, :] * sum_v_inv) * v_bc
        if FULL_VHAT:
            # full per-element 2nd moment (rank-1 mis-shapes the conv v̂; see FULL_VHAT note).
            if self.v_full is None:
                self.v_full = torch.zeros(N, K, dtype=torch.float32, device=dev)
            self.v_full = beta2 * self.v_full + (1 - beta2) * g2
            vhat = self.v_full * v_bc

        scale_fwd = _scale_fwd(self.row_exp, self.col_exp, mantissa_bias)
        scale_inv = 1.0 / scale_fwd

        e_H, e_L, s_slow, v_slow = (x.to(torch.int32) for x in unpack_dual(self.packed_w))

        # (A) per-arm velocity; SHARED gap (CARRY = 1).
        d_fs_L = e_L.to(torch.float32)
        d_fs_H = e_H.to(torch.float32)
        d_sv = (s_slow.to(torch.float32) - v_slow.to(torch.float32)) * CARRY_FLAT

        # (B) per-arm coherence.
        coh_L, coh_raw_L = compute_coherence(d_fs_L, d_sv, vhat, drift_cancel_C, coh_kappa,
                                             sum_v_inv, N, K, scale_fwd, use_coh_vhat)
        coh_H, coh_raw_H = compute_coherence(d_fs_H, d_sv, vhat, drift_cancel_C, coh_kappa,
                                             sum_v_inv, N, K, scale_fwd, use_coh_vhat)

        # (C) e-weighted blended raw coherence.
        wL = d_fs_L.abs(); wH = d_fs_H.abs(); wsum = wL + wH
        coh_raw_blend = torch.where(wsum > 0, (coh_raw_L * wL + coh_raw_H * wH) / (wsum + 1e-30),
                                    0.5 * (coh_raw_L + coh_raw_H))

        # (D) preconditioned gradient inflow into each arm.
        fine_v = (e_L + e_H).to(torch.float32)
        noise_w = (fine_v - drift_cancel_C * d_sv) * scale_fwd
        v_proxy = noise_w * noise_w * v_scale
        denom = torch.pow(v_proxy + eps, precond_p)
        if not two_grad:
            delta = -lr * (grad_W / denom).clamp(-step_cap, step_cap) * scale_inv
            inflow_L = 0.5 * delta
            inflow_H = 0.5 * delta
        else:
            inflow_L = -lr * (grad_W / denom).clamp(-step_cap, step_cap) * scale_inv
            inflow_H = -lr * (grad_W_H / denom).clamp(-step_cap, step_cap) * scale_inv

        # (E) intent = velocity + inflow -> SR new e.
        intent_L = d_fs_L + inflow_L
        intent_H = d_fs_H + inflow_H
        if beta1 != 0.0:
            # beta1 = coherence-gated VELOCITY amplification (NOT Adam's grad-EMA). Gate on
            # coh_RAW, not the cf-discounted coh (PB:1016 comment: feeding cf-coh in "broke the
            # noise-cancellation and diverged" -- coh >= coh_raw over-amplifies). It compounds;
            # stable only for beta1*coh <~ chase rate (~alpha), so keep beta1 small (<= ~0.1).
            intent_L = intent_L + consf * beta1 * coh_raw_L * d_fs_L
            intent_H = intent_H + consf * beta1 * coh_raw_H * d_fs_H
        e_L = _sr_round(intent_L, e_L, pos, salt ^ 0x000000A1)
        e_H = _sr_round(intent_H, e_H, pos, salt ^ 0x000000B2)

        # (F) chase, gain 1, AFFINE coh gate -- EXACT same-scale transfer (CARRY = 1).
        gate_L = chase_floor + (1.0 - chase_floor) * coh_L
        gate_H = chase_floor + (1.0 - chase_floor) * coh_H
        chase_L = alpha * gate_L * e_L.to(torch.float32) * consf
        chase_H = alpha * gate_H * e_H.to(torch.float32) * consf
        tick_L = _sr_round(chase_L / float(CARRY_FLAT), e_L, pos, salt ^ 0x5A5A5A5A)
        tick_H = _sr_round(chase_H / float(CARRY_FLAT), e_H, pos, salt ^ 0xA5A5A5A5)
        s_slow = s_slow + tick_L + tick_H
        e_L = e_L - tick_L * CARRY_FLAT
        e_H = e_H - tick_H * CARRY_FLAT

        # (G) per-arm bracketed evaporation off the blended raw coherence (build-gated).
        if gf_consol > 0.0:
            lam = lr * gf_consol
            lam_L = lam * (1.0 - self.bracket_d)
            lam_H = lam * (1.0 + self.bracket_d)
            one_minus = 1.0 - coh_raw_blend
            ef_L = torch.clamp(lam_L * one_minus, max=1.0 - min_leak)
            ef_H = torch.clamp(lam_H * one_minus, max=1.0 - min_leak)
            res_L = e_L.to(torch.float32); res_H = e_H.to(torch.float32)
            p_build_L = torch.clamp(res_L.abs() / (evap_build_min + 1e-30), max=1.0)
            p_build_H = torch.clamp(res_H.abs() / (evap_build_min + 1e-30), max=1.0)
            ok_L = (_hash_uniform(e_L, pos, salt ^ 0x42420001) < p_build_L).to(torch.float32)
            ok_H = (_hash_uniform(e_H, pos, salt ^ 0x42420002) < p_build_H).to(torch.float32)
            e_L = e_L - _sr_round(ef_L * res_L * ok_L * consf, e_L, pos, salt ^ 0x0E0E0001)
            e_H = e_H - _sr_round(ef_H * res_H * ok_H * consf, e_H, pos, salt ^ 0x0E0E0002)

        # (H) leak s_slow -> v_slow (the anchor), blended-coh floor, EXACT (CARRY = 1).
        coh_bar = 0.5 * (coh_L + coh_H)
        gap = (s_slow.to(torch.float32) - v_slow.to(torch.float32))
        dv = alpha_v_fast * gap * consf * (leak_floor + (1.0 - leak_floor) * coh_bar)
        tick_v = _sr_round(dv, e_L, pos, salt ^ 0x33335555)
        v_slow_new = torch.clamp(v_slow + tick_v, INT8_MIN, INT8_MAX)
        if mass_preserve:
            s_slow = s_slow - (v_slow_new - v_slow)
        v_slow = v_slow_new
        # NO deploy dissipation: only the fast (e_L/e_H) evaporates; (s_slow+v_slow) persists.

        # (J) re-exponent safety (flat: SR-halve all four + bump row_exp on saturation).
        e_H, e_L, s_slow, v_slow, self.row_exp = renormalize_flat(
            e_H, e_L, s_slow, v_slow, self.row_exp, pos, salt)

        # (K) clamp + repack.
        e_H = e_H.clamp(INT8_MIN, INT8_MAX); e_L = e_L.clamp(INT8_MIN, INT8_MAX)
        s_slow = s_slow.clamp(INT8_MIN, INT8_MAX); v_slow = v_slow.clamp(INT8_MIN, INT8_MAX)
        self.packed_w = pack_dual(e_H, e_L, s_slow, v_slow)

        return {
            "coh_L": float(coh_L.mean()), "coh_H": float(coh_H.mean()),
            "coh_raw_blend": float(coh_raw_blend.mean()),
            "e_L_abs_mean": float(e_L.abs().to(torch.float32).mean()),
            "e_H_abs_mean": float(e_H.abs().to(torch.float32).mean()),
            "s_slow_abs_mean": float(s_slow.abs().to(torch.float32).mean()),
            "v_slow_abs_mean": float(v_slow.abs().to(torch.float32).mean()),
        }
