#!/usr/bin/env bash
# Ratio-coherence A/B (enwik8, validated recipe, lr5e-4, 5k):
#   rc_gate   = 64-bit coh_pre gate  -> must reproduce ~1.1438 (kernel-safety + anchor)
#   rc_ratio  = 32-bit ratio-coh (gate chase+leak by live coh, drop coh_pre) = EXPERIMENT
#   rc_nogate = no gate              -> lower anchor (~1.17)
# Q: does the 32-bit ratio-coh match the 64-bit gate (free gate, parsimony restored)?
set -u
cd /c/concord
BENCH="--mode concord --data nanogpt_data/enwik8 --max_iters 5000 --eval_interval 500 --eval_iters 50 --seed 0 --data_seed 1234 --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --reb_stats"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## ratio-coh $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm rc_gate    --coh_gate
run_arm rc_ratio   --ratio_coh
run_arm rc_nogate
echo "######## RATIO-COH AB DONE ########"
