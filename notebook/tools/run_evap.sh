#!/usr/bin/env bash
# EVAPORATION sweep (option 3, user's term): evap = lr*gf_consol*(1-coh)*d_fs, d_fs=s_fast
# (kernel L568). rho_eff = lr*gf_consol; at lr=5e-4, gf_consol=100 -> rho_eff=0.05/step on
# INCOHERENT coords (coherent ~0 via the (1-coh) factor). Goal: drain the 57.8% stranded
# s_fast so ratio-coh's slow path fills and sv becomes deployable (target: near gate-sv 1.53).
# +min-floor (option 2) for comparison + base smooth-gate. eval_interval 500 (halve overhead).
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 500 --eval_iters 50 --seed 0 --data_seed 1234 --watch_accum"
CONS="--eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## evap $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
# (3) evaporation at the right scale (rho_eff = lr*gf_consol)
run_arm ce_evap100  $CONC $CONS --ratio_coh --gf_consol 100   # rho_eff ~0.05/step (incoherent)
run_arm ce_evap200  $CONC $CONS --ratio_coh --gf_consol 200   # rho_eff ~0.10/step (incoherent)
# (2) min floors comparison: chase floored 0.1 (s_fast drains via chase, not evap)
run_arm ce_minfloor $CONC $CONS --ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1
# base recipe + smooth-gate deploy-slow during training
run_arm fg_base     $CONC --fast_gain_anneal
echo "######## EVAP DONE ########"
