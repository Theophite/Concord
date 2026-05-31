#!/usr/bin/env bash
# Concord diffusion lr sweep BELOW 5e-4. The {5e-4,1e-3,2e-3} sweep was monotone
# (5e-4=0.0352 < 1e-3=0.0381 < 2e-3=0.0478) -> optimum may be <5e-4. Maps the
# bottom of the U. (enwik8 analog: 3e-4 UNDER-trained at 5k, so expect a turn.)
# Same toy DDPM setup: ch128 bsz128 5000 iters, shared init + data_seed, coh_gate.
# Anchors: Concord 5e-4=0.0352, AdamW best (lr1e-3)=0.0321.
set -u
cd /c/concord
BENCH="--mode concord --max_iters 5000 --eval_interval 500 --eval_iters 20 --ch 128 --bsz 128 --seed 0 --data_seed 1234 --warmup_iters 100 --coh_gate"
run_arm() {
  local lr="$1" tag="$2"
  for a in 1 2 3; do
    echo "######## diff-lr $tag lr=$lr (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_diffusion.py $BENCH --concord_lr "$lr" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm 3e-4 d_conc_3e4
run_arm 2e-4 d_conc_2e4
run_arm 1e-4 d_conc_1e4
echo "######## DIFFUSION LR-DOWN DONE ########"
