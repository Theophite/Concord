#!/usr/bin/env bash
# ortho_k50 ONLY: orthogonalize d_sv every 50 steps. 3000 iters (floor ~2000 + margin).
# vs base ref (best-over-run): live 1.579 / sv 1.526 / v 1.549. WIN: beat sv 1.526 OR
# v 1.549 by >0.005. Tests if REAL d_sv is full-rank enough for NS5 to help (probe_muon3
# warned low-rank d_sv -> NS5 sprays energy into null space -> HURTS).
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 3000 --eval_interval 250 --eval_iters 100 --seed 0 --data_seed 1234 --eval_consolidated --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate --ortho_slow_every 50"
for a in 1 2 3 4 5; do
  echo "######## ortho_k50 (attempt $a) ########"
  [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
  python src/train_nanogpt.py $B --tag ortho_k50 && break
  echo "[ortho_k50] retry"; sleep 15
done
echo "######## ORTHOARM DONE ########"
