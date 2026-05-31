#!/usr/bin/env bash
# lr sweep BELOW 1e-3 for v-hat + fixed coherence gate. 2e-3 overshoots (2.44),
# 1e-3 is stable (1.1601) -> 1e-3 is near the cliff, true optimum likely lower.
# Same init+batch order. vs 1e-3=1.1601, Adam=1.0686.
set -u
cd /c/concord
COMMON="--mode concord --data nanogpt_data/enwik8 --max_iters 5000 \
  --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 \
  --v_scale 0 --gf_trust_delta_sq 1 --precond_p 0.5 --alpha 0.1 --step_cap 10 \
  --eps 1e-10 --rebalance_every 1 --coh_gate"
run_arm() {
  local lr="$1" tag="$2"
  for a in 1 2 3; do
    echo "######## coh-gate-lrdown $tag lr=$lr (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $COMMON --concord_lr "$lr" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm 7e-4 cglr_7e4
run_arm 5e-4 cglr_5e4
run_arm 3e-4 cglr_3e4
echo "######## COH-GATE LR-DOWN RUNS DONE ########"
