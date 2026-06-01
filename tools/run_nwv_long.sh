#!/usr/bin/env bash
# LONG-HORIZON nwv test, FIRED on the dev box (crash-retry wrapper for the flaky PSU;
# no resume -> a retry restarts the arm from iter 0). 20k iters, 2 arms.
# HYPOTHESIS: nwv inert at 5k because horizon-starved (best_val ~1500 < nwv-active ~2000).
# At 20k, v-hat-mature-then-reshape (delay 2000 + ramp 2000 -> full by 4000) gets ~16k
# steps of runway. WIN: long_nwv best live OR deployed sv beats long_base by >0.005.
set -u
cd /c/concord
COMMON="--data nanogpt_data/input.txt --max_iters 20000 --eval_interval 500 --eval_iters 100 --seed 0 --data_seed 1234 --eval_consolidated --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3 4 5; do
    echo "######## long $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $COMMON "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
  echo "[$tag] GAVE UP after 5 attempts"
}
run_arm long_base
run_arm long_nwv --noise_weighted_v --nwv_beta 1.0 --nwv_delay 2000 --nwv_beta_warmup 2000
echo "######## NWVLONG DONE ########"
