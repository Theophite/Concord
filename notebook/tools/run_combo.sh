#!/usr/bin/env bash
# COMBO: split the difference between minfloor (consolidate: chase floored -> MOVES s_fast
# to s_slow, lossless but banks noise too) and evaporation (drain: deletes incoherent
# s_fast, lossy but noise-only). On a borderline coord they compete; the split lets the
# FLOOR bank the borderline-coherent while EVAP removes the clearly-incoherent.
# Reference points already in hand: minfloor (floor 0.1, evap 0) + evap100/200 (floor 0,
# evap on). These two add the (floor>0, evap>0) interior.
# Read: --watch_accum (s_fast share back to ~5%? v_slow healthy?) + deploy sv vs gate-sv 1.53.
set -u
cd /c/concord
BENCH="--data nanogpt_data/input.txt --max_iters 5000 --eval_interval 500 --eval_iters 50 --seed 0 --data_seed 1234 --watch_accum"
CONS="--eval_consolidated"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## combo $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
# literal midpoint: half the minfloor floor (0.05) + half the evap100 rate (gf_consol 50)
run_arm cb_split  $CONC $CONS --ratio_coh --ratio_chase_floor_min 0.05 --ratio_leak_floor_min 0.05 --gf_consol 50
# consolidation-leaning: keep minfloor's full floor (0.1) + light drain (gf_consol 50)
run_arm cb_consol $CONC $CONS --ratio_coh --ratio_chase_floor_min 0.1 --ratio_leak_floor_min 0.1 --gf_consol 50
echo "######## COMBO DONE ########"
