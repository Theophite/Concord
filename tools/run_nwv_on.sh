#!/usr/bin/env bash
# NOISE-WEIGHTED v-hat, the (1-coh) arm ONLY (baseline nwv_base already locked best
# live 1.5838 = the known gate ref 1.584 / sv 1.530). Tests gating the rank-1 v-hat
# accumulation on (1-coh): v-hat fits NOISE power -> sqrt(v) small on coherent coords
# -> RELATIVELY bigger steps there (correctly-signed precond; cwv used the inverted
# sign=coh and was null). Win = beats baseline best live 1.5838 / sv 1.530 by >~0.005.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## nwv $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm nwv_on  $CONC --noise_weighted_v
echo "######## NWV_ON DONE ########"
