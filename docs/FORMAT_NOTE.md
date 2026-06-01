# Concord packed-B: the storage format is block-float, NOT "int8"

Correcting a framing error repeated throughout the 2026-05-31 session ("the int8 cascade",
"int8 SR rounding destroys X"). The cascade is **not int8**. It is the **mantissa of a
block-floating-point number** whose exponent is shared per row+column. The int8 fields are
the *high bits of one integer mantissa*, not the quantization grain.

## What's actually stored (verified: prototype_packed_b.py L126-134)

One int32 per weight, unpacked as:
```
s_fast    = packed >> 16            # int16  (bits 31:16)  -- low/fine bits of the mantissa
s_slow_i8 = (packed << 16) >> 24    # int8   (bits 15:8)   -- mid bits  (carries x128)
v_slow_i8 = (packed << 24) >> 24    # int8   (bits  7:0)   -- high/anchor bits (carries x128)
m_eff     = s_slow_i8*128 + s_fast + v_slow_i8*128         # a SINGLE signed integer mantissa
weight    = m_eff * 2^(row_exp + col_exp - mantissa_bias)  # block-float: shared row+col exponent
```

## The key correction

- **m_eff is one ~17-bit signed integer mantissa.** s_fast (int16, ±32767) supplies the
  fine/low bits; s_slow and v_slow (int8, ×128) supply the coarse/high bits. They are
  CONCATENATED into one number, not three independent int8 quantities. The realizable
  magnitude is ~|s_slow+v_slow|*128 + s_fast, i.e. up to ~17 bits of signed mantissa.
- **The exponent is shared per (row, col): block floating point.** row_exp + col_exp give
  a per-row and per-column power-of-two envelope (the rebalance ratchet retunes it). So a
  weight is mantissa(17-bit) x 2^(shared exponent) -- a block-float scalar.
- **Therefore the per-weight precision is FINER than bf16, not coarser.** bf16 = 1 sign + 8
  exponent + 7 mantissa (8-bit effective mantissa with implicit leading 1). Concord = ~17-bit
  mantissa on a *shared* exponent. The mantissa resolution is higher than bf16's; what's
  given up is the PER-ELEMENT exponent (bf16 has one per value; Concord shares it per
  row+col). That is the actual storage tradeoff: more mantissa bits, fewer exponent bits
  (amortized over a row/col block).

## Why the "int8" framing caused real errors this session

- **Muon-chase "sub-quantum" analysis (probe_muon):** I treated the SR quantum as 128
  mantissa units ("int8 grain") and concluded the chase tick was sub-quantum. The chase
  tick IS sub-quantum relative to the int8 s_slow field, but that's because the FINE bits
  live in s_fast (int16); the rounding granularity of the full m_eff is ~1 mantissa unit,
  not 128. The s_slow->v_slow transfers are the coarse part; the live weight is finer. The
  qualitative conclusions held, but "int8 cascade" mis-locates where precision lives.
- **General:** any statement like "int8 SR rounding limits X" should be "the SR transfer
  between the COARSE (int8x128) accumulators is at 128-mantissa grain, but the live weight
  m_eff and the s_fast velocity are ~17-bit / 1-unit grain on a block-float exponent."

## Correct one-liners (use these going forward)

- Storage: **32 bits/param = a 17-bit signed mantissa (s_fast:s_slow:v_slow concatenated)
  on a shared per-row+col block-float exponent.** Not "int8."
- Tradeoff vs bf16: **more mantissa (17 vs 8), shared exponent (1 per row+col vs 1 per
  element).** Higher value-resolution, coarser dynamic-range locality.
- The cascade (chase/leak) moves mass between the COARSE high-bit accumulators (s_slow,
  v_slow at x128) via SR; the FINE bits (s_fast, int16) carry the live velocity at full
  resolution. "int8" only describes the two coarse fields, never the live weight.

## Related (logged in CONTROL_PLANE.md)

- The slow cascade reconstructs a beta1=0.9 gradient MOMENTUM via difference-of-EMAs
  (d_sv = s_slow - v_slow; align +0.87 with true EMA_0.9(grad)). Concord is already a
  momentum optimizer; the two accumulators at different leak rates ARE a momentum buffer.
- RANK (corrected -- "intrinsic rank ~35" was WRONG, see below): the gradient/momentum is
  LOW-RANK on tiny-shakespeare but it is TASK- and LAYER-specific, NOT a Concord property.
  Real-data measurement (probe_rank.py, real nanoGPT + tiny-shakespeare, true grad_W, @it120,
  momentum eff-rank as participation-ratio / SVs-for-90%-energy, of min-dim 384):
      attn.c_attn  PR 40 / r90 16     attn.c_proj  PR 19 / r90 7     mlp.c_fc  PR 77 / r90 36
  So low-rank holds HERE (tens of 384), but with a ~5x spread across layer types, and this is
  vocab-65 tiny-shakespeare (highly compressible). A richer task (the SDXL target) would push
  rank UP. The earlier "~35/384" was a probe_muon6 ARTIFACT (that probe used a rank-32
  SYNTHETIC target -> measured its own construction). Do NOT cite a single intrinsic rank.
- WHY a modest shared-exponent mantissa suffices, and WHY deploy-slow works, are CONSISTENT
  with low-rank-on-this-task but NOT proven to depend on it -- treat as hypotheses, not facts.
- MUON verdict is TASK-SCOPED, not architectural: on tiny-shakespeare the momentum is low-rank
  (above) and NS5 inflates it well past its support (probe_muon6: rank ~35->280 ON THE
  SYNTHETIC probe; real layers r90 7-36 -> NS5 would similarly over-spread), consistent with
  the measured +0.2 harm. But on a HIGHER-rank task the rank argument weakens and Muon is
  UNTESTED. Correct statement: "Muon hurts on this low-rank task," NOT "Muon is the wrong
  operator for Concord."
- deploy-slow: ship consolidated_weight() = (s_slow+v_slow)*128*2^exp (drop s_fast); beats
  the live m_eff weight by ~0.04-0.05 val nats, scale-stable.
