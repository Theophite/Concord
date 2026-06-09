# Concord × OneTrainer — status & working notes (SDXL full-UNet finetune, 24 GB)

**This integration is BUILT and in production — it trains SDXL today.** Read
[`DESIGN.md`](DESIGN.md) for the as-built architecture (file:line into the real tree). This file
is the status + how-to-work-on-it. (Prior revisions described it as a from-scratch build plan for
a "next bot" and pointed at a recon-scratch clone — that was wrong; corrected here.)

---

## 0. Constraints (do not violate)
- **Filesystem rule:** only touch the session subtree, `C:\concord\`, and `C:\fisher\`.
- **GPU contention:** the user runs ComfyUI (SDXL app) on the same card; it grabs ~17 GB. Check
  `nvidia-smi --query-gpu=memory.free` before any GPU run; ASK the user to free it rather than
  killing ComfyUI.
- **Don't over-ask / prefer doing.** Architecture is settled and shipped; improve, don't re-survey.

## 1. Where everything is (two trees, one remote `Theophite/Concord.git`)
- **Optimizer library** — this repo (`C:\concord`), `src/`. The importable modules:
  `prototype_packed_b.py` (the kernel: packed AdamW, lazy gate, fused packed matmul),
  `concord_winner.py` (winner config + `swap_unet_to_winner`/`swap_text_encoder_to_anchor` +
  `winner_step` schedule + `GatedRebalance`), `concord_embedding{,_packed}.py` (norm-preserving
  new-token embeddings), `control_plane.py`/`token_init.py` (per-token routing + init),
  `vhat_buckets.py` (per-aspect-ratio v̂ cache). `src/` mirrors the fork's
  `modules/util/optimizer/concord/` 1:1.
- **OneTrainer integration** — fork at `C:\fisher\OneTrainer-clean`, branch **`concord-integration`**
  (unrelated history to this repo's `main`). Glue NOT in the library: `concord_ot.py`
  (`ConcordController` — swap driver, per-step hooks, save-consolidation, packed-embedding setup),
  `concord_graph.py` (`ManualUNetGraph` CUDA-graph, opt-in), plus the minimal hooks in
  `create.py` (`Optimizer.CONCORD` → aux SGD), `StableDiffusionXLFineTuneSetup.py`, and
  `StableDiffusionXLModelSaver.py`. See DESIGN.md §1–7 for exact file:line.
- The `C:\concord\OneTrainer/` clone here is **recon scratch** (gitignored, no integration code).

## 2. What's built & validated
- [x] Full UNet swap (794 nn.Linear/Conv2d → packed), self-stepping in backward, aux SGD.
- [x] Full SDXL finetune runs end-to-end and saves a standard sharded UNet (consolidate-on-save).
- [x] TE frozen-anchor training + packed new-token embeddings (textual inversion), reversible save.
- [x] Gradient accumulation (~0 extra mem), bf16, non-reentrant gradient checkpointing.
- [x] Lazy-update gate: no-op-safe on dense training (ot_noop A/B, OFF==ON at τ=1e-4, 2026-06-09).
- [x] Fused packed matmul kernels: numerics match cuBLAS (fwd exact, bwd within bf16); the active
      fwd/bwd path. Compile-validated on torch 2.5.1/triton 3.1.0 **and** 2.9.1/triton 3.5.1.

## 3. How to run a training
- Env: the fork's venv (`C:\fisher\OneTrainer-clean\venv`, torch 2.9.1). Pick `Optimizer.CONCORD`
  in the OneTrainer config; bf16 + latent caching + gradient checkpointing.
- Knobs (DESIGN.md §7): `gf_consol`, `noise`/`sigmag_peak`, `ratio_coh`, `lazy_gate`/
  `lazy_active_thresh`; `concord_te_anchor`, `concord_packed_embeddings`, `concord_fused_matmul`
  (on by default), `concord_cuda_graph` (off/experimental).
- The validated recipe is baked in `concord_winner.py : WINNER`; spec in
  [`WINNING_CONFIG.md`](../WINNING_CONFIG.md).

## 4. The recipe (the validated winner)
Authoritative spec: [`WINNING_CONFIG.md`](../WINNING_CONFIG.md). A **fluctuation–dissipation pair**
on the bare rank-1 v̂ AdamW (32 b/param):
- **Dissipation** (the "split", confirmed): `ratio_coh` + chase/leak floors 0.1 + `gf_consol 50`.
  Same-seed nanoGPT A/B: deploy-sv **1.518** vs bare 1.540.
- **Fluctuation** (noise): isotropic `sigmag_peak ≈ 0.6`. σ-sweep on the split: **1.4967** (−0.021).
  Full pair **1.497 vs AdamW 1.534 / Muon 1.578** at 32 b/param.
Deploy off `consolidated_weight()`. Use isotropic noise, never Σ_g (ablation: iso ≥ Σ_g).

## 5. Open items
- **CUDA-graph v1** (`make_graphed_callables`) NaNs on the first step → use `ManualUNetGraph` (v2)
  or eager (the default). v1 is opt-in behind `concord_cuda_graph`.
- **Noise σ magnitude** is single-seed / nanoGPT-only (trajectory jitter ≈ half the −0.021 effect)
  → multi-seed on SDXL before treating the magnitude as load-bearing. The *mechanism* is done.
- **EMA** over param-less swapped layers: if EMA is enabled, confirm it reads the deploy weight or
  is disabled for swapped layers (the weight is a buffer, not a Parameter).
- **Library sync** (this PR): `src/` now matches the production fork; keep them in lockstep on
  future kernel changes (edit `src/`, mirror into the fork, or vice-versa).

## 6. Anti-patterns / lessons (don't repeat)
- **Don't claim a CUDA graph is correct without the eager-vs-graph SR-floor gate** (a real bug
  GROWS; SR noise stays within a floor and converges).
- **Don't fabricate measurements** — run the probe/test, read the real output, THEN write the number.
- **Validate across Triton versions before shipping kernel changes** — the lazy-gate addition made
  a gate condition a 3-way `or` chain that newer Triton accepts but ≤3.1 rejects; the fix was to
  parenthesize into pairwise `or`. Compile-check on the repo's declared `torch>=2.1`, not just the
  dev env.
- **Two trees, one remote** — the optimizer library (`src/`) and the OneTrainer fork's
  `concord/` package must stay in lockstep; a kernel edit in one is a latent bug until mirrored.
