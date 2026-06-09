# Concord in the literature — an annotated bibliography

How the pieces of Concord relate to prior work, organized along the design itself: the
packed format, the memory-system argument for it, the update rule, and the reasons it
works. Each section names its nearest neighbors and states where Concord diverges. §5
states what we could *not* find.

**Provenance caveat:** compiled from model knowledge (cutoff January 2026), not from a
systematic literature search. ArXiv IDs are given only where confident; verify
everything before citing externally. Treat §5's novelty claim as "no known prior art,"
not "proven absence."

---

## 1. The packed format

Concord stores weight and optimizer state in one int32 per parameter — three integer
fields at different timescales (int16 `s_fast`, int8 `s_slow`, int8 `v_slow`) on shared
per-row/per-column exponents — with stochastic rounding keeping low-bit accumulation
unbiased. Every ingredient has prior art; the *composition* does not.

**Quantized optimizer state.** Dettmers et al. [8] (8-bit) and Li et al. [9] (4-bit)
compress Adam's moment tensors with block-wise quantization. This is the conservative
version of the idea: the state remains *separate* from the weight, just smaller, and is
dequantized/requantized around an otherwise standard update. Concord's departure: there
is no separate state to compress — the moments are implicit in the gaps between the
weight's own accumulators (§4).

**Factored / sublinear state.** AdaFactor [3] replaces the [N,K] second moment with
row/col marginals; SM3 [10] generalizes factored adaptivity to cover sets. Concord uses
AdaFactor's factorization directly for v̂ — and re-applies the same trick to *dynamic
range* (per-row + per-col exponents), which we have not seen elsewhere, though the
primitive is standard block floating point.

**Block floating point / shared exponents.** Hybrid BFP training [13], the Microscaling
(MX) formats [14], and bfloat16 studies [12] establish shared-exponent integer mantissas
as a viable training representation. Concord's twist: the shared exponent is factored
(row + col) rather than per-block, and the mantissa is *partitioned into a timescale
cascade* rather than being one number.

**Stochastic rounding.** Gupta et al. [11] is the canonical reference for SR making
low-precision accumulation unbiased; mixed-precision training [15] established the
master-copy alternative Concord exists to delete. Concord leans on SR harder than most:
five independent SR streams per element (gradient tick, chase, leak, two anchored-decay
terms) are the only thing standing between a 17-bit mantissa and accumulation bias —
and sub-LSB gradient signal survives *because* of SR, not despite it.

**Quantized weights for fine-tuning.** QLoRA [16] freezes 4-bit base weights and trains
fp16 adapters — quantization as a way to *avoid* training the weights. Concord trains
the quantized representation itself, full-rank.

## 2. Why packed: HBM traffic and in-register computation

The systems argument is the optimizer-step analogue of IO-aware kernel design.

**The memory wall.** Arithmetic is cheap; moving bytes is expensive — by orders of
magnitude in energy [19] — and most deep-learning kernels are bandwidth-bound in the
roofline sense [18]. FlashAttention [17] is the modern exemplar: restructure the
computation so intermediate state never round-trips through HBM. Concord applies the
same logic to the *optimizer*: the persistent state is one 4-byte word per parameter,
and the entire read–gate–precondition–step–consolidate–repack pipeline runs in
registers between one load and one store (`_apply_packed_adamw_kernel`; see
`HOW_IT_WORKS.md` §5–6).

Approximate per-parameter, per-step optimizer traffic (beyond the unavoidable
activation/gradient flow): mixed-precision AdamW touches the bf16 weight, fp32 master,
and two fp32 moments — roughly 30 B of reads+writes; Concord touches the int32 word and
emits the next forward's bf16 weight — roughly 12 B, with the AdaFactor vectors O(N+K)
amortized to ~0. Resident optimizer state drops from ~14–16 B/param to 4. On SDXL this
is the difference between LoRA-only and a ~15 GB full-UNet fine-tune.

**Fused into backward.** LOMO [20] and AdaLOMO [21] are the closest in *systems shape*:
fuse the parameter update into the backward pass so gradients and optimizer state never
materialize tensor-wide (AdaLOMO ≈ LOMO + AdaFactor — the nearest published neighbor to
Concord's goal). The difference: LOMO-family optimizers compute a standard update in a
fused sweep over normal-precision state; Concord's layers are self-stepping autograd
Functions over packed integer state, with the dequantization fused *inside* the matmul
(no persistent bf16 weight at all) and the whole step capturable into a CUDA graph.
Kernel infrastructure: Triton [22]; gradient checkpointing [23] composes with the
self-step because the optimizer write happens only in the (single) gradient-producing
backward.

## 3. The update rule

The simplest form (one weight; `W = P + u`):

```text
g̃   = g + σ‖g‖ξ                      # fluctuation
v̂   = AdaFactor-EMA(g̃²)              # rank-1 preconditioner
coh = μ²/(μ²+(u−μ)²),  μ = C*·d      # Wiener gain from the timescale telescope
u  ← u − lr·clip(g̃/√(v̂+ε), ±c) − lr·κ(1−coh)·u + β1·coh·u
P  ← P + α·gc·u,   u ← (1−α·gc)·u    # continuous Lookahead
```

Component-by-component neighbors:

- **Preconditioner**: Adam/AdamW [1, 2] define the class; AdaFactor [3] supplies the
  rank-1 v̂ and the update-clipping ancestor of `step_cap`; Adagrad [4] the lineage.
  `precond_p` (partial adaptivity) is Padam's knob [5].
- **Consolidation**: Lookahead [6] is the chase, run every step instead of every k and
  gated by coherence. Shipping `P` is Polyak–Ruppert averaging [24] / SWA [25]; in
  diffusion practice, EMA weights are standard and post-hoc EMA [26] shows how much the
  averaging horizon matters — Concord gets the average for free, *inside* the same
  32 bits.
- **Minimal-state direction**: signSGD [27] and Lion [28] showed how little state
  adaptive-ish training needs (one buffer); Concord continues to zero explicit buffers.
  The opposite pole — more state for better curvature (Shampoo [29], and Muon [30] on
  the same bench) — is what the winner was benchmarked against.
- **Two timescales of the same signal**: AdEMAMix [7] keeps fast+slow gradient EMAs but
  *adds* them for the direction; Concord *subtracts* them to estimate SNR. Same
  structure, opposite use.
- **Variance as deviation-from-prediction**: AdaBelief [31] tracks Var(g − m) — "belief"
  in the gradient — as a stored EMA. Concord's `u − C*·d` is the same conceptual
  quantity (innovation residual) read out of state that already exists.
- **Kalman/Wiener machinery**: the gain `coh = S²/(S²+N²)` is Wiener's estimator [33] /
  the steady-state Kalman gain [32]. Kalman-filter optimizers exist (KOALA [34];
  Vuckovic [35]; Ollivier's identification of online natural gradient with Kalman
  filtering [36]) but carry explicit covariance state; Concord's gain costs zero bytes
  because the telescope is made of fields the format already stores. C\* is the
  analytically derived drift-cancellation making the innovation zero-mean under pure
  drift (`compute_drift_cancel_C`; corrected for the mass-preserving leak in
  `RESULTS.md` §2).

## 4. Why it works

**(a) Adaptivity without stored moments.** Adam's second moment estimates gradient
scale/variance; AdaBelief showed the *useful* part is variance of the deviation from
the predictable component [31]. Concord's claim, validated on its benches: the gaps
between accumulators at different timescales carry that same information. The fast–slow
gap is a drift-lagged velocity; the slow–anchor gap is a drift telescope; their
calibrated ratio is a per-weight SNR. Nothing else we know of extracts moments from the
*storage format*.

**(b) The fluctuation–dissipation pair is Langevin machinery.** Injected gradient noise
plus friction is discretized Langevin dynamics; SGLD [37] made the sampling reading
precise, Mandt–Hoffman–Blei [38] made constant-lr SGD-as-inference precise, and Yaida
[39] derived exact fluctuation–dissipation relations for SGD (and proposed using them
adaptively — a spiritual ancestor of the κ autotuner). Gradient-noise injection as
regularization is classic [40]; the gradient-noise-scale measurement [41] is the
ancestor of the probe. Concord's specific configuration — isotropic noise *through* the
preconditioner, friction gated per-weight by measured SNR — and the measured
consequence (the dissipation curve κ\*(noise), the memorization-budget property;
`RESULTS.md` §4) are, to our knowledge, its own.

**(c) Shipping the mean, not the sample.** Iterate averaging approximates the posterior
mean of the SGD stationary distribution [38, 24]; SWA [25] and diffusion-EMA practice
[26] are the applied versions. Concord's deploy weight is exactly this, and the
measured deploy-vs-live gap appears precisely when gradients are noisy — consistent
with the theory's prediction.

**(d) The anchor is a prior.** Decay toward pretrained weights rather than zero is
L2-SP [42]; weighting that pull by parameter importance is EWC [43]; the
variational/natural-gradient reading of optimizers-as-inference is Khan et al.
[44, 45]. Concord's `v_slow` (initialized to the pretrained weight in fine-tuning) is
the prior mean, `wd_anchor` its precision, and the per-weight confidence weighting
falls out of the same telescope — no Fisher estimate stored.

**(e) Why dissipation suppresses memorization.** Networks fit patterns before noise
[46, 47]; wrong-label gradients are individually weak, mutually incoherent early, and
only consolidate through slow persistent drift. A friction that drains *incoherent*
velocity therefore taxes noise-fitting specifically — measured: bare 43% wrong-label
memorization vs 24% with dissipation at equal accuracy cost elsewhere
(`RESULTS.md` §3–4). The same mechanism explains the one honest limit: memorization
drift that has become coherent passes the gate, which is why β1 (a coherence
amplifier) helps only on clean streams (`RESULTS.md` §6).

## 5. What we could not find

The unclaimed core, stated precisely: **a training algorithm in which the quantized
weight representation itself is the optimizer state — integer accumulators at different
timescales sharing one machine word, whose pairwise gaps are the adaptive moments.**
Neighbors approach from every side — compressed-but-separate state [8, 9], factored
state [3, 10], fused-into-backward updates [20, 21], two-timescale EMAs [7],
innovation-based variance [31], Kalman gains [34–36] — but each keeps the optimizer's
memory distinct from the weight's representation. The intersection (implicit state ×
full rank × number-format co-design × in-register self-stepping) appears to be
unoccupied. Structural reasons it stayed empty: the field's memory-saving energy went
to low-rank methods (LoRA [48], GaLore [49]); the low-bit-optimizer line deliberately
stayed drop-in; and the idea requires simultaneous fluency in optimization theory and
kernel-level numerics, which is a thin intersection.

If prior art exists, it most likely hides under keywords like "implicit optimizer
state," "self-quantizing training," "multi-timescale integer accumulators," or in the
analog/neuromorphic literature (multi-timescale synaptic state is a known motif in
computational neuroscience — cascade models of synaptic plasticity are a conceptual
cousin worth checking, though they are not gradient-based optimizers).

---

## References

IDs only where confident; verify before external use.

1. Kingma, Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980
2. Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101
3. Shazeer, Stern. *Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.* ICML 2018. arXiv:1804.04235
4. Duchi, Hazan, Singer. *Adaptive Subgradient Methods…* JMLR 2011
5. Chen, Gu. *Closing the Generalization Gap of Adaptive Gradient Methods…* (Padam) arXiv:1806.06763
6. Zhang, Lucas, Hinton, Ba. *Lookahead Optimizer: k steps forward, 1 step back.* NeurIPS 2019. arXiv:1907.08610
7. Pagliardini, Ablin, Grangier. *The AdEMAMix Optimizer: Better, Faster, Older.* arXiv 2024
8. Dettmers, Lewis, Shleifer, Zettlemoyer. *8-bit Optimizers via Block-wise Quantization.* ICLR 2022. arXiv:2110.02861
9. Li, Chen, Zhu. *Memory Efficient Optimizers with 4-bit States.* NeurIPS 2023
10. Anil, Gupta, Koren, Singer. *Memory-Efficient Adaptive Optimization.* (SM3) NeurIPS 2019. arXiv:1901.11150
11. Gupta, Agrawal, Gopalakrishnan, Narayanan. *Deep Learning with Limited Numerical Precision.* ICML 2015. arXiv:1502.02551
12. Kalamkar et al. *A Study of BFLOAT16 for Deep Learning Training.* arXiv 2019
13. Drumond, Lin, Jaggi, Falsafi. *Training DNNs with Hybrid Block Floating Point.* NeurIPS 2018
14. Rouhani et al. *Microscaling Data Formats for Deep Learning.* arXiv 2023 (OCP MX specification)
15. Micikevicius et al. *Mixed Precision Training.* ICLR 2018. arXiv:1710.03740
16. Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314
17. Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135
18. Williams, Waterman, Patterson. *Roofline: An Insightful Visual Performance Model…* CACM 2009
19. Horowitz. *Computing's Energy Problem (and what we can do about it).* ISSCC 2014
20. Lv et al. *Full Parameter Fine-tuning for Large Language Models with Limited Resources.* (LOMO) arXiv 2023
21. Lv et al. *AdaLomo: Low-memory Optimization with Adaptive Learning Rate.* arXiv 2023
22. Tillet, Kung, Cox. *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations.* MAPL 2019
23. Chen, Xu, Zhang, Guestrin. *Training Deep Nets with Sublinear Memory Cost.* arXiv:1604.06174
24. Polyak, Juditsky. *Acceleration of Stochastic Approximation by Averaging.* SIAM J. Control Optim. 1992
25. Izmailov et al. *Averaging Weights Leads to Wider Optima and Better Generalization.* (SWA) UAI 2018. arXiv:1803.05407
26. Karras et al. *Analyzing and Improving the Training Dynamics of Diffusion Models.* (EDM2 / post-hoc EMA) CVPR 2024
27. Bernstein et al. *signSGD: Compressed Optimisation for Non-Convex Problems.* ICML 2018. arXiv:1802.04434
28. Chen et al. *Symbolic Discovery of Optimization Algorithms.* (Lion) NeurIPS 2023. arXiv:2302.06675
29. Gupta, Koren, Singer. *Shampoo: Preconditioned Stochastic Tensor Optimization.* ICML 2018. arXiv:1802.09568
30. Jordan et al. *Muon.* Technical report / blog, 2024
31. Zhuang et al. *AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients.* NeurIPS 2020. arXiv:2010.07468
32. Kalman. *A New Approach to Linear Filtering and Prediction Problems.* J. Basic Eng. 1960
33. Wiener. *Extrapolation, Interpolation, and Smoothing of Stationary Time Series.* MIT Press 1949
34. Davtyan et al. *KOALA: A Kalman Optimization Algorithm with Loss Adaptivity.* AAAI 2022
35. Vuckovic. *Kalman Gradient Descent: Adaptive Variance Reduction in Stochastic Optimization.* arXiv 2018
36. Ollivier. *Online Natural Gradient as a Kalman Filter.* Electron. J. Stat. 2018
37. Welling, Teh. *Bayesian Learning via Stochastic Gradient Langevin Dynamics.* ICML 2011
38. Mandt, Hoffman, Blei. *Stochastic Gradient Descent as Approximate Bayesian Inference.* JMLR 2017. arXiv:1704.04289
39. Yaida. *Fluctuation-Dissipation Relations for Stochastic Gradient Descent.* ICLR 2019
40. Neelakantan et al. *Adding Gradient Noise Improves Learning for Very Deep Networks.* arXiv:1511.06807
41. McCandlish, Kaplan, Amodei et al. *An Empirical Model of Large-Batch Training.* arXiv:1812.06162
42. Li, Grandvalet, Davoine. *Explicit Inductive Bias for Transfer Learning with Convolutional Networks.* (L2-SP) ICML 2018
43. Kirkpatrick et al. *Overcoming Catastrophic Forgetting in Neural Networks.* (EWC) PNAS 2017. arXiv:1612.00796
44. Khan et al. *Fast and Scalable Bayesian Deep Learning by Weight-Perturbation in Adam.* (Vadam) ICML 2018
45. Khan, Rue. *The Bayesian Learning Rule.* JMLR 2023
46. Zhang, Bengio, Hardt, Recht, Vinyals. *Understanding Deep Learning Requires Rethinking Generalization.* ICLR 2017. arXiv:1611.03530
47. Arpit et al. *A Closer Look at Memorization in Deep Networks.* ICML 2017. arXiv:1706.05394
48. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685
49. Zhao et al. *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.* ICML 2024
