#!/usr/bin/env bash
# NOISE-WEIGHTED v-hat, the CORRECT cosine-in (per user): HOLD beta=0 first so the rank-1
# v-hat EMA accumulates a CLEAN estimate AND the coh ratio resolves, THEN cosine beta 0->1.
# The prior ramp-from-0 ran concurrently with v-hat buildup (reshaped a forming v-hat) ->
# inert. Caveat: v-hat maturity (~1000) + ratio resolution (~1000) ~ best_val timing (~1500)
# on this short run, so the clean-first window may leave little room before overfit.
# Baseline (gate, no nwv): best live 1.584 / sv 1.530. Win = beat by >0.005.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate --noise_weighted_v --nwv_beta 1.0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## nwvcos $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm nc_d1000_w1000  $CONC --nwv_delay 1000 --nwv_beta_warmup 1000   # clean to 1000, ramp 1000-2000
run_arm nc_d1500_w500   $CONC --nwv_delay 1500 --nwv_beta_warmup 500    # clean to 1500 (past best_val), fast ramp
echo "######## NWVCOS DONE ########"
