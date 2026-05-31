#!/usr/bin/env bash
# How much of the SGD->Adam gap does a RANK-1 v-hat (Adafactor factorization)
# capture? FactoredAdam = AdamW with v-hat factored to rank-1 on 2D weights;
# everything else identical to the AdamW baseline. Same init + batch order
# (seed 0, data_seed 1234) -> lands directly comparable on the SGD-chase(1.43)
# <-> Adam(1.07) axis. lr=1e-3 matched to Adam + 2e-3 in case rank-1 wants more.
set -u
cd /c/concord
OUT=C:/concord/compare_out
COMMON="--mode factored --data nanogpt_data/enwik8 --max_iters 5000 \
  --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --weight_decay 0.1"

run_arm() {
  local lr="$1" tag="$2" save="$3"
  for a in 1 2 3; do
    if [ -n "$save" ] && [ -f "${save}_factored.pt" ]; then echo "[$tag] done"; return 0; fi
    echo "######## factored $tag lr=$lr (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    if [ -n "$save" ]; then
      python src/train_nanogpt.py $COMMON --factored_lr "$lr" --tag "$tag" --save_prefix "$save"
      [ -f "${save}_factored.pt" ] && return 0
    else
      python src/train_nanogpt.py $COMMON --factored_lr "$lr" --tag "$tag" && return 0
    fi
    echo "[$tag] retry"; sleep 15
  done
}

run_arm 1e-3 e8f_lr1e3 "$OUT/e8fact"
run_arm 2e-3 e8f_lr2e3 ""
echo "######## FACTORED RUNS DONE ########"
