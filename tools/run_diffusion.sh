#!/usr/bin/env bash
# Toy DDPM (CIFAR-10, ~2.5M tiny UNet) Concord vs AdamW, eps-prediction. Shared
# init + data_seed => identical (image, timestep, noise) draws from step 0. Both
# wd=0 (diffusion convention; wd shown neutral on enwik8). 5000 iters, ch=128,
# bsz=128, cosine over 5k. Small lr sweep each side -- diffusion's lr optimum may
# differ from enwik8's. Headline pair (adamw1e-3, conc5e-4) first. ~4 min/arm.
set -u
cd /c/concord
BENCH="--max_iters 5000 --eval_interval 500 --eval_iters 20 --ch 128 --bsz 128 --seed 0 --data_seed 1234 --warmup_iters 100"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## diffusion $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_diffusion.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm d_adamw_1e3  --mode adamw   --adamw_lr 1e-3 --weight_decay 0
run_arm d_conc_5e4   --mode concord --concord_lr 5e-4 --coh_gate
run_arm d_adamw_2e4  --mode adamw   --adamw_lr 2e-4 --weight_decay 0
run_arm d_conc_1e3   --mode concord --concord_lr 1e-3 --coh_gate
run_arm d_conc_2e3   --mode concord --concord_lr 2e-3 --coh_gate
echo "######## DIFFUSION SWEEP DONE ########"
