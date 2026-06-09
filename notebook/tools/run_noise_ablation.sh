#!/usr/bin/env bash
# NOISE ABLATION on the CONFIRMED split baseline (ratio+consol, deploy-sv 1.518).
# Isolates which of the doc's 3 noise ingredients carry the effect:
#   base   : split, NO noise, lr_min_frac 0.1   -> the confirmed baseline
#   floor  : split, NO noise, lr_min_frac 0.2   -> does raising the LR floor ALONE help?
#   full   : split + Sigma_g rising + floor 0.2 -> the faithful recipe
#   iso    : split + ISOTROPIC rising + floor   -> does Sigma_g SHAPING matter? (doc: iso shuts gate)
#   const  : split + Sigma_g CONSTANT + floor   -> does RISING-LATE matter vs constant?
# Metric: best deployed sv (consolidated_weight). Win = beats base sv by >0.005 AND beats floor.
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
SPLIT="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $B "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm na_base   $SPLIT --lr_min_frac 0.1
run_arm na_floor  $SPLIT --lr_min_frac 0.2
run_arm na_full   $SPLIT --lr_min_frac 0.2 --sigmag 0.3
run_arm na_iso    $SPLIT --lr_min_frac 0.2 --sigmag 0.3 --sigmag_iso
run_arm na_const  $SPLIT --lr_min_frac 0.2 --sigmag 0.3 --sigmag_const
echo "######## NOISE-ABLATION DONE ########"
