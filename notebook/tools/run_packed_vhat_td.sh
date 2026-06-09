#!/usr/bin/env bash
# Packed rank-1 v-hat WITH tick-down re-enabled (bidirectional exponent
# rebalance: tick-up on saturation, tick-down to reclaim precision when int8
# headroom allows). Tests: (1) does tick-down improve the lr=1e-3 winner
# (1.1712, precision recovery)? (2) does it rescue higher lr that previously
# ran away (lr=3e-3 was 2.10 via tick-up ratchet runaway)? Same init + batch
# order (seed 0, data_seed 1234). v_scale=0 + gf_trust=1 => denom=(v_hat+eps)^.5.
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
    echo "######## packed-vhat-TD $tag lr=$lr (attempt $a) ########"
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

run_arm 1e-3 ptd_l1e3 "$OUT/e8pvhat_td"   # the 1.17 config, now with tick-down
run_arm 2e-3 ptd_l2e3 ""                   # mid lr
run_arm 3e-3 ptd_l3e3 ""                   # the runaway lr (was 2.10) -- rescued?
echo "######## PACKED VHAT TICK-DOWN RUNS DONE ########"
