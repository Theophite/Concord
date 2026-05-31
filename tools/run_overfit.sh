#!/usr/bin/env bash
# OVERFITTING regime (capacity >> data): the 10M-param GPT on tiny-shakespeare (1.1MB)
# -> strong overfit. THE decisive test for Concord's noise-resistance mechanisms: does
# the coherence gate's "structurally can't fit noise" yield a real GENERALIZATION win
# (lower BEST val + smaller final-vs-best overfit rise) vs AdamW and Concord-no-gate,
# which should overfit harder? All wd=0 (isolate the optimizer's IMPLICIT reg), same
# init + data_seed. Read: best_val (early-stop min) and final_val (post-overfit).
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## overfit $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm of_adamw   --mode adamw --adamw_lr 1e-3 --weight_decay 0   # overfit baseline (no reg)
run_arm of_nogate  $CONC                                           # Concord, no gate (control)
run_arm of_gate    $CONC --coh_gate                                # coherence gate (64-bit)
run_arm of_ratio   $CONC --ratio_coh                               # ratio-coh gate (32-bit)
echo "######## OVERFIT DONE ########"
