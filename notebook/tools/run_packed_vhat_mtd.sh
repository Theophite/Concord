#!/usr/bin/env bash
# Packed rank-1 v-hat with ASYMMETRIC tick-down (median-gated): tick-UP eager
# (max>MAX_M), tick-DOWN lazy (per-row/col median |int8| <= 31 = bulk underflow;
# outliers clip). Kills the growth-phase churn that made the naive max-gated
# tick-down WORSE (1.26 vs 1.17 no-td). Same init+batch order. Target: beat 1.17.
set -u
cd /c/concord
OUT=C:/concord/compare_out
COMMON="--mode concord --data nanogpt_data/enwik8 --max_iters 5000 \
  --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 \
  --v_scale 0 --gf_trust_delta_sq 1 --precond_p 0.5 --alpha 0.1 --step_cap 10 \
  --eps 1e-10 --rebalance_every 1"

run_arm() {
  local lr="$1" tag="$2" save="$3"
  for a in 1 2 3; do
    if [ -n "$save" ] && [ -f "${save}_concord.pt" ]; then echo "[$tag] done"; return 0; fi
    echo "######## packed-vhat-MTD $tag lr=$lr (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    if [ -n "$save" ]; then
      python src/train_nanogpt.py $COMMON --concord_lr "$lr" --tag "$tag" --save_prefix "$save"
      [ -f "${save}_concord.pt" ] && return 0
    else
      python src/train_nanogpt.py $COMMON --concord_lr "$lr" --tag "$tag" && return 0
    fi
    echo "[$tag] retry"; sleep 15
  done
}

run_arm 1e-3 mtd_l1e3 "$OUT/e8pvhat_mtd"   # vs no-td 1.1712 / naive-td 1.2612
run_arm 2e-3 mtd_l2e3 ""                    # does lazy-down open a faster-lr regime?
echo "######## PACKED VHAT MEDIAN-TICK-DOWN RUNS DONE ########"
