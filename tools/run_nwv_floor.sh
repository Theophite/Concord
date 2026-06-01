#!/usr/bin/env bash
# NOISE-WEIGHTED v-hat with a FLOORED curve: w = 1 - beta*coh, floored at (1-beta).
# Diagnosis of the failed raw run (beta=1, best live 1.598 > baseline 1.584): the
# (1-coh) curve drives w->0 as coords mature -> UNBOUNDED LR boost late -> overfits.
# Fix = floor the curve (same lesson as the cascade floors). beta<1 bounds the boost.
# References: gate baseline live 1.584 / sv 1.530; unbounded beta=1 live 1.598.
# Win = a floored arm beats 1.584 (best live) / 1.530 (sv) by >~0.005.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate --noise_weighted_v"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## nwvf $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm nwvf_b50  $CONC --nwv_beta 0.5    # floor 0.50, LR boost cap ~1.41x
run_arm nwvf_b25  $CONC --nwv_beta 0.25   # floor 0.75, LR boost cap ~1.15x (gentler)
echo "######## NWVF DONE ########"
