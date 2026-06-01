#!/usr/bin/env bash
# Muon-at-slow<->v_slow A/B (overfit bench). orthogonalize d_sv=s_slow-v_slow every K=50
# steps (probe_muon2: K~50 is where SR-quantized orthogonality survives; ~leak timescale).
# Base ref (no ortho, same bench, best-over-run): live 1.579 / sv 1.526 / v 1.549 (@~it2000).
# WIN: ortho beats sv 1.526 OR v 1.549 by >0.005. (Tests if REAL d_sv is full-rank enough
# for NS5 to help -- probe_muon3 warned low-rank d_sv would HURT.) 8000 iters (floor ~2000
# + margin). Two arms in one wrapper for an apples-to-apples same-schedule compare.
set -u
cd /c/concord
B="--data nanogpt_data/input.txt --max_iters 8000 --eval_interval 250 --eval_iters 100 --seed 0 --data_seed 1234 --eval_consolidated --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0 --coh_gate"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3 4 5; do
    echo "######## $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $B "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm ortho_base
run_arm ortho_k50 --ortho_slow_every 50
echo "######## ORTHO DONE ########"
