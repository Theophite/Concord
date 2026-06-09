#!/usr/bin/env bash
# SIGMA SWEEP on the winning ISOTROPIC noise arm (na_iso sigma=0.3 -> sv 1.5095, best in
# the ablation). Deterministic at fixed seed (na_base==det_b proven), so differences are
# real. Find the magnitude optimum vs base sv=1.5180. Rising-late + lr_min_frac 0.2 + iso,
# matching na_iso; only sigma_peak varies. (LR floor 0.2 alone hurt to 1.5243 -> any arm
# beating 1.5180 means noise is doing real work over the floor it requires.)
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
ISO="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50 --lr_min_frac 0.2 --sigmag_iso"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $B "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm ss_010 $ISO --sigmag 0.1
run_arm ss_020 $ISO --sigmag 0.2
run_arm ss_050 $ISO --sigmag 0.5
run_arm ss_070 $ISO --sigmag 0.7
echo "######## SIGMA-SWEEP DONE ########"
