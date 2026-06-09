#!/usr/bin/env bash
# Confirm the rank-ladder finding AT Concord's converged solution (not just at
# init). Re-run Concord (deterministic; adds aux_final = embeddings+LN so the
# full fp32 model can be rebuilt at the solution), then probe per-direction
# gradient SNR along dC / D / random dirs at the solution AND re-confirm at init.
set -u
cd /c/concord
OUT=C:/concord/compare_out
PREFIX="$OUT/e8gap"

for a in 1 2 3; do
  python -c "import torch;d=torch.load('${PREFIX}_concord.pt',weights_only=False);exit(0 if 'aux_final' in d else 1)" 2>/dev/null && \
    { echo "[solprobe] concord save already has aux_final"; break; }
  echo "######## re-run concord w/ aux save (attempt $a) ########"
  [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
  python src/train_nanogpt.py --mode concord --data nanogpt_data/enwik8 \
    --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 \
    --concord_lr 0.05 --eps 1.0 --save_prefix "$PREFIX" --tag e8g_concord_aux
  sleep 5
done

echo "==== PROBE @ Concord solution ===="
python src/noise_rank_probe.py --prefix "$PREFIX" --K 96 --r 16 --at concord
echo ""
echo "==== PROBE @ init (re-confirm, same settings) ===="
python src/noise_rank_probe.py --prefix "$PREFIX" --K 96 --r 16 --at init
echo "######## SOLUTION PROBE DONE ########"
