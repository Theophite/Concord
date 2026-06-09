#!/usr/bin/env bash
# Gate-schedule diagnostic: put the cosine on the GATE (gate_gain), hold LR CONSTANT,
# and watch the adjusted-gradient magnitude (median per-step |dW|) curve -- vs EXACTLY
# the same seed with constant LR and NO gate. Isolates whether scheduling the gate
# shapes the effective per-coordinate movement (and where), decoupled from lr decay.
# Both: v-hat (v_scale=0,gf_trust=1,eps1e-10), const lr=5e-4, same init+batch order.
set -u
cd /c/concord
COMMON="--mode concord --data nanogpt_data/enwik8 --max_iters 5000 \
  --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 \
  --v_scale 0 --gf_trust_delta_sq 1 --precond_p 0.5 --alpha 0.1 --step_cap 10 \
  --eps 1e-10 --rebalance_every 1 --concord_lr 5e-4 --const_lr --watch_dw"
run_arm() {
  local tag="$1"; shift; local extra="$*"
  for a in 1 2 3; do
    echo "######## gate-sched $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $COMMON $extra --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm gs_gatecos --coh_gate --gate_cosine   # cosine ON the gate, const lr
run_arm gs_nogate                              # no gate, const lr (same seed)
echo "######## GATE-SCHED RUNS DONE ########"
