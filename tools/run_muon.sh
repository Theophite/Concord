#!/usr/bin/env bash
# THE CONTROL: does REAL Muon (native fp, faithful NS5+Nesterov, standard 2D-split) beat
# AdamW on tiny-shakespeare overfit? This contextualizes the whole Concord-orthogonalization
# thread: if Muon ~= or < AdamW here, the low-rank-task story is confirmed and orthogonalization
# is correctly closed for this bench. If Muon >> AdamW, my CASCADE integration was the problem,
# not Muon. Refs (same bench): AdamW best 1.534; Concord deploy-sv 1.526. wd=0, shared seed.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm mu_adamw  --mode adamw --adamw_lr 1e-3 --weight_decay 0
run_arm mu_muon   --mode muon  --muon_lr 0.02  --aux_lr 3e-3 --weight_decay 0
echo "######## MUON-AB DONE ########"
