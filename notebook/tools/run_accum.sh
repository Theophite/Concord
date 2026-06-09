#!/usr/bin/env bash
# Where does the mass sit at the overfit endpoint? Re-run the 3 Concord arms from the
# overfit harness with --watch_accum. Hypothesis under test: if the gate refuses to chase
# incoherent (noise) coords into s_slow, it STRANDS that mass in s_fast -- which is part
# of m_eff (= s_slow*128 + s_fast + v_slow*128), so the noise is still in the deployed
# weight. That would explain why the gate does NOT reduce overfit. Predict: gate/ratio
# show a HIGHER s_fast share than no-gate.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 1000 --eval_iters 50 --seed 0 --data_seed 1234 --watch_accum"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## accum $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm ac_nogate  $CONC
run_arm ac_gate    $CONC --coh_gate
run_arm ac_ratio   $CONC --ratio_coh
echo "######## ACCUM DONE ########"
