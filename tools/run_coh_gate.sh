#!/usr/bin/env bash
# FIXED coherence gate on the working packed v-hat: Wiener coh=S/(S+noise^2) gates
# commitment (freeze incoherent/stuck coords), on top of the rank-1 v-hat. Same
# init+batch order. vs no-gate baseline 1.1712 (and Adam 1.0686). The gate damps
# commitment of the ~96% incoherent coords, so also probe a hotter lr.
set -u
cd /c/concord
OUT=C:/concord/compare_out
COMMON="--mode concord --data nanogpt_data/enwik8 --max_iters 5000 \
  --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 \
  --v_scale 0 --gf_trust_delta_sq 1 --precond_p 0.5 --alpha 0.1 --step_cap 10 \
  --eps 1e-10 --rebalance_every 1 --coh_gate"

run_arm() {
  local lr="$1" tag="$2" save="$3"
  for a in 1 2 3; do
    if [ -n "$save" ] && [ -f "${save}_concord.pt" ]; then echo "[$tag] done"; return 0; fi
    echo "######## coh-gate $tag lr=$lr (attempt $a) ########"
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

run_arm 1e-3 cg_l1e3 "$OUT/e8cg"   # matched to the 1.17 no-gate baseline
run_arm 2e-3 cg_l2e3 ""             # hotter (gate damps commitment)
echo "######## COH-GATE RUNS DONE ########"
