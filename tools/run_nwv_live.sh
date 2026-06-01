#!/usr/bin/env bash
# NOISE-WEIGHTED v-hat, LIVE ratio-coh version (no coh_pre buffer; bootstrap-correct:
# coh=0 at init -> w=1 = normal v-hat). Fixes the broken coh_pre=1-init version (which
# starved v-hat for ~1000 iters: nwv best 1.598 vs base 1.584). w=1-beta*coh from the
# packed s_fast:s_slow:v_slow ratio -> v-hat fits NOISE power -> bigger steps on coherent
# coords. References: gate baseline best live 1.584 / sv 1.530.
# Two gate configs x two betas. nwv now works WITHOUT --coh_gate (reads the ratio direct).
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## nwvlive $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
# with the fixed gate (comparable to the 1.584 baseline)
run_arm nl_gate_b50  $CONC --coh_gate --noise_weighted_v --nwv_beta 0.5
run_arm nl_gate_b100 $CONC --coh_gate --noise_weighted_v --nwv_beta 1.0
# bare layer (no coh_gate): nwv from the ratio alone -- the true 32-bit parsimony config
run_arm nl_bare_b50  $CONC --noise_weighted_v --nwv_beta 0.5
echo "######## NWVLIVE DONE ########"
