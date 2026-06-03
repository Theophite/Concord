#!/usr/bin/env bash
# SETTLE the tentative-best: shipped bare recipe (nogate) vs the split config (+consol),
# SAME seed, same bench, deployed-sv the metric. (FIX: $CONC was defined-but-unused in v1,
# so both arms ran default eps=1.0 SGD-chase, NOT the validated v-hat recipe -- now passed.)
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm ab_nogate $CONC
run_arm ab_consol $CONC --ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50
echo "######## SPLIT-AB DONE ########"
