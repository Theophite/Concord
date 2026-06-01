#!/usr/bin/env bash
# long_nwv ONLY, 8000 iters (covers the nwv-active window: delay 2000 + ramp 2000 = full
# beta by iter4000, plus margin; long_base's floor landed at iter2000 so 8k is plenty).
# Compare to long_base: best live 1.5792 / sv 1.5260 / v 1.5485 (@iter ~2000). WIN: nwv
# beats sv 1.526 OR v 1.549 by >0.005.
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 8000 --eval_interval 250 --eval_iters 100 --seed 0 --data_seed 1234 --eval_consolidated --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate --noise_weighted_v --nwv_beta 1.0 --nwv_delay 2000 --nwv_beta_warmup 2000"
for a in 1 2 3 4 5; do
  echo "######## long_nwv (attempt $a) ########"
  [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
  python src/train_nanogpt.py $B --tag long_nwv && break
  echo "[long_nwv] retry"; sleep 15
done
echo "######## NWVARM DONE ########"
