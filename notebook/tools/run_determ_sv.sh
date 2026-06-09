#!/usr/bin/env bash
# Is best-deployed-sv DETERMINISTIC at fixed seed? Rerun the SAME config (na_base = split,
# no noise) TWICE, identical seed. If best-sv is bit-identical, the ablation's ~0.009
# orderings are REAL signal, not noise -> noise genuinely helps. If they vary by ~0.01,
# the orderings are within run noise -> can't rank the noise arms.
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated"
SPLIT="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50 --lr_min_frac 0.1"
python src/train_nanogpt.py $B $SPLIT --tag det_a
python src/train_nanogpt.py $B $SPLIT --tag det_b
echo "######## DETERM DONE ########"
