#!/usr/bin/env bash
# Re-run just the 20k diffusion pair (the 10k pair already landed: Concord 0.0331,
# AdamW 0.0315, gap halved 0.0031->0.0016). First 20k attempt wedged the flaky box
# at iter 14000 (Concord val was 0.0322 there, still improving). Retry both 20k arms.
set -u
cd /c/concord
COMMON="--max_iters 20000 --eval_interval 1000 --eval_iters 20 --ch 128 --bsz 128 --seed 0 --data_seed 1234 --warmup_iters 100"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3 4; do
    echo "######## dh20 $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_diffusion.py $COMMON "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm dh_conc_20k  --mode concord --concord_lr 5e-4 --coh_gate
run_arm dh_adamw_20k --mode adamw   --adamw_lr 1e-3 --weight_decay 0
echo "######## DH20 DONE ########"
