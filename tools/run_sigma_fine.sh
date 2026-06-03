#!/usr/bin/env bash
# FINE sigma grid to resolve the band shape. Known (deterministic): 0.0=1.5180, 0.1=1.5216,
# 0.2=1.5097, 0.3=1.5095, 0.5=1.5158, 0.7=1.5098. Fill 0.25/0.35/0.4/0.6: is [0.2,0.7] truly
# FLAT (~1.509) or is 0.3 a real peak + 0.5 a real dip? Same config as the sweep (iso, rising,
# floor 0.2). Deterministic, so the grid resolves real structure.
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
run_arm sf_025 $ISO --sigmag 0.25
run_arm sf_035 $ISO --sigmag 0.35
run_arm sf_040 $ISO --sigmag 0.40
run_arm sf_060 $ISO --sigmag 0.60
echo "######## SIGMA-FINE DONE ########"
