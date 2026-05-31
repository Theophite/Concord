#!/usr/bin/env bash
# DEPLOYED-WEIGHT test (overfit regime, tiny-shakespeare). If the gate strands noise in
# s_fast, then the weight you DEPLOY should drop s_fast. Per-variant best val for:
#   live = s_slow*128 + s_fast + v_slow*128   (what the overfit run measured)
#   sv   = (s_slow + v_slow)*128              (exactly m_eff minus s_fast)
#   s2v  = (s_slow + 2*v_slow)*128            (double the long ANCHOR; user's 2x-v_slow)
#   2v / 2s = 2*v_slow*128 / 2*s_slow*128     (legacy single-accumulator)
# Q: under --coh_gate, does sv or s2v beat the live gate best (1.584) AND the no-gate
# live best (1.566)? i.e. does cutting s_fast recover (or exceed) the no-gate weight?
# +--watch_accum so each arm is self-contained (where the mass sits).
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 --eval_consolidated --watch_accum"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## cons $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm cv_nogate  $CONC
run_arm cv_gate    $CONC --coh_gate
run_arm cv_ratio   $CONC --ratio_coh
echo "######## CONS-EVAL DONE ########"
