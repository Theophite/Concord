#!/usr/bin/env bash
# LR-TAIL fix for noise-weighted v-hat (principled lever, not the gate shape).
# Diagnosis: nwv (1-coh, beta=1) gives coherent coords bigger RELATIVE steps -- good
# EARLY (it led at iter 1500) -- but the ABSOLUTE LR stays too hot LATE, so those boosted
# steps overshoot in the overfit phase (best live 1.598 vs gate 1.584; beta-floor only
# half-recovered it -> wrong lever). Fix: let the GATE do relative shaping (beta=1), let
# the COSINE do absolute annealing -> drop the tail (lr_min_frac 0.1 -> 0.0).
# 2x2 design (rows known): default-tail no-nwv 1.5838 | default-tail nwv 1.598;
# this run adds low-tail no-nwv (CONTROL: isolates the tail effect) + low-tail nwv (TEST).
# nwv adds value IFF lt_nwv < lt_ctrl by >~0.005 (else the tail just helped everyone).
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated --lr_min_frac 0.0"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## lrtail $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm lt_ctrl  $CONC                       # gate, tail->0, NO nwv (isolates the tail effect)
run_arm lt_nwv   $CONC --noise_weighted_v    # gate, tail->0, + nwv beta=1 (test)
echo "######## LRTAIL DONE ########"
