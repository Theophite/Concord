#!/usr/bin/env bash
# Does the toy-diffusion Concord-vs-AdamW gap close with horizon, like enwik8's
# did (0.077->0.037->0.023, halving per doubling)? 5k anchors: Concord 0.0352
# (lr5e-4), AdamW 0.0321 (lr1e-3), gap 0.0031. Run both at 10k and 20k, cosine
# spanning each, same init + data_seed. 10k pair FIRST (early read).
# NOTE: CIFAR diffusion epochs = iters*128/50000 -> 5k~13ep, 10k~26ep, 20k~51ep.
# eps-pred's random-(t,noise) augmentation slows overfit (train~=val held at 5k);
# we report train AND val to catch any divergence at the longer horizons.
set -u
cd /c/concord
COMMON="--eval_interval 1000 --eval_iters 20 --ch 128 --bsz 128 --seed 0 --data_seed 1234 --warmup_iters 100"
run_arm() {
  local tag="$1" iters="$2"; shift 2
  for a in 1 2 3; do
    echo "######## diff-horizon $tag ($iters) (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_diffusion.py --max_iters "$iters" $COMMON "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm dh_conc_10k  10000 --mode concord --concord_lr 5e-4 --coh_gate
run_arm dh_adamw_10k 10000 --mode adamw   --adamw_lr 1e-3 --weight_decay 0
run_arm dh_conc_20k  20000 --mode concord --concord_lr 5e-4 --coh_gate
run_arm dh_adamw_20k 20000 --mode adamw   --adamw_lr 1e-3 --weight_decay 0
echo "######## DIFFUSION HORIZON DONE ########"
