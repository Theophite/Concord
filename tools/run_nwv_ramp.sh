#!/usr/bin/env bash
# NOISE-WEIGHTED v-hat with COSINE-RAMPED beta (the "cosine it in" fix). Live ratio-coh
# is quantization-noise early (tiny s_slow/v_slow ints ~first 1000 iters) -> no-ramp
# live-coh tracked behind base + worse floor (b50 best ~1.602 vs base 1.584). Ramp beta
# 0->target over warmup so noisy early coh can't corrupt v-hat until the ratio resolves.
# STOPPING BAR: nwv is a win ONLY if a ramped arm beats base 1.584 (live) or 1.530 (sv)
# by >0.005; else nwv is INERT here (4th negative) -> revert the nwv edits.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate --noise_weighted_v"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## nwvramp $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm nr_w1000_b50  $CONC --nwv_beta 0.5 --nwv_beta_warmup 1000
run_arm nr_w1000_b100 $CONC --nwv_beta 1.0 --nwv_beta_warmup 1000
run_arm nr_w2000_b100 $CONC --nwv_beta 1.0 --nwv_beta_warmup 2000
echo "######## NWVRAMP DONE ########"
