#!/usr/bin/env bash
# Correctness + speed gate: eager vs --cuda_graph, same seed, 400 iters, no eval noise
# (eval_interval huge so timing reflects train steps). Concord is bit-deterministic
# (probe_determ) so graphed MUST match eager to ~1e-3 if capture is correct.
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 400 --eval_interval 100 --eval_iters 50 --seed 0 --data_seed 1234 --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --coh_gate"
echo "===== EAGER ====="; python src/train_nanogpt.py $B --tag ck_eager
echo "===== GRAPH ====="; python src/train_nanogpt.py $B --cuda_graph --tag ck_graph
echo "===== GRAPHCHECK DONE ====="
