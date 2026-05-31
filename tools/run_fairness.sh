#!/usr/bin/env bash
# Fairness sweep on the 0.075-nat residual (best packed 1.1438 vs AdamW 1.0686).
# The regularization mismatch: the AdamW baseline ran wd=0.1; Concord ran wd=0.
# Dropout=0 for both (matched), and enwik8 is data>>capacity so dropout can't help.
# Build the wd 2x2 (each optimizer at its own tuned lr; one variable = wd):
#   AdamW  wd=0    (A0)   vs  Concord wd=0   (have 1.1438)  -> FAIR gap at wd=0
#   Concord wd>0  (C1/C2) vs  AdamW   wd=0.1 (have 1.0686)  -> FAIR gap at wd=0.1
# Arm A0.1 reproduces the AdamW baseline under THIS box's conditions (~0.02 run var)
# so the wd effect is measured apples-to-apples, not vs a stale prior-session number.
# Same init + batch order (seed 0, data_seed 1234), 5000 iter. Decisive arm first.
set -u
cd /c/concord
BENCH="--data nanogpt_data/enwik8 --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --concord_lr 5e-4 --coh_gate"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## fairness $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm fair_adamw_wd0    --mode adamw --adamw_lr 1e-3 --weight_decay 0      # decisive control
run_arm fair_adamw_wd0p1  --mode adamw --adamw_lr 1e-3 --weight_decay 0.1    # reproduce baseline (same box)
run_arm fair_conc_wd0p1   $CONC --concord_wd 0.1                              # Concord matched wd
run_arm fair_conc_wd0p2   $CONC --concord_wd 0.2                              # lr-scaled wd (Concord lr=5e-4 vs Adam 1e-3)
echo "######## FAIRNESS SWEEP DONE ########"
