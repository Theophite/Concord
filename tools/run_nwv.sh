#!/usr/bin/env bash
# NOISE-WEIGHTED v-hat A/B (overfit regime, tiny-shakespeare). Tests gating the rank-1
# v-hat ACCUMULATION on (1-coh): v-hat fits NOISE power, so sqrt(v) is small on coherent
# coords -> RELATIVELY bigger steps there. This is the correctly-signed coherence
# preconditioner (cwv tested the inverted sign = coh, and was neutral/stripped).
# Baseline = the 64-bit fixed gate (small-model ref: best live 1.584 / sv 1.530).
# Read: best live + best sv (deployed). Win = nwv_on beats nwv_base by >~0.005 (else null).
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
run_arm nwv_base  $CONC                       # 64-bit fixed gate baseline
run_arm nwv_on    $CONC --noise_weighted_v    # + (1-coh)-weighted v-hat
echo "######## NWV DONE ########"
