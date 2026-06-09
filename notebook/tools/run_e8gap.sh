#!/usr/bin/env bash
# enwik8 AdamW-vs-(unwhitened)Concord COMPARABILITY run: identical init (seed 0)
# + identical batch order (data_seed 1234, dedicated generator). Each optimizer
# at its locked-best enwik8 config (Concord eps=1/lr=0.05 -> 1.4436; AdamW
# lr=1e-3/wd=0.1 -> 1.0707). Saves init+final per-Linear weights for compare_gap.
# Auto-retries (unstable box). The {prefix}_{mode}.pt file is the success marker.
set -u
cd /c/concord
OUT=C:/concord/compare_out
mkdir -p "$OUT"
PREFIX="$OUT/e8gap"
COMMON="--data nanogpt_data/enwik8 --max_iters 5000 --eval_interval 250 \
  --eval_iters 50 --seed 0 --data_seed 1234 --save_prefix $PREFIX"

run_mode () {
  local mode="$1"; shift
  local extra="$*"
  for attempt in 1 2 3 4; do
    if [ -f "${PREFIX}_${mode}.pt" ]; then
      echo "[$mode] already done -> ${PREFIX}_${mode}.pt"; return 0
    fi
    echo "######## e8gap $mode attempt $attempt ########"
    [ "$attempt" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py --mode "$mode" $COMMON $extra --tag "e8g_$mode"
    [ -f "${PREFIX}_${mode}.pt" ] && { echo "[$mode] OK"; return 0; }
    echo "[$mode] crashed/no-save; retrying"; sleep 15
  done
  echo "[$mode] FAILED after retries"; return 1
}

run_mode concord --concord_lr 0.05 --eps 1.0
run_mode adamw   --adamw_lr 1e-3 --weight_decay 0.1
echo "######## E8 GAP COMPARABILITY RUNS DONE ########"
