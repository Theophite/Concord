#!/usr/bin/env bash
# Run BOTH best-Concord and the AdamW baseline out to 10k iters (2x), cosine
# spanning the full 10k (lr_at keys off max_iters), same init + batch order
# (seed 0, data_seed 1234). Question: is the ~0.077-nat residual PERMANENT, or
# does Concord's slow-but-selective descent keep closing it past 5k? Must extend
# AdamW too -- Concord-10k vs AdamW-5k would be unfair (2x the compute).
# 5k refs: Concord 1.1438, AdamW ~1.067. Concord arm first (the question's subject).
set -u
cd /c/concord
BENCH="--data nanogpt_data/enwik8 --max_iters 10000 --eval_interval 500 --eval_iters 50 --seed 0 --data_seed 1234"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## longhorizon $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm lh_concord_10k --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --concord_lr 5e-4 --coh_gate --eval_consolidated
run_arm lh_adamw_10k   --mode adamw --adamw_lr 1e-3 --weight_decay 0.1
echo "######## LONGHORIZON DONE ########"
