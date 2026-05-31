#!/usr/bin/env bash
# BIGGER nanoGPT (n_embd=640, n_layer=10, n_head=10 -> ~49M params, 4.5x the 10.78M small)
# on tiny-shakespeare (1.1MB): capacity>>data is even MORE extreme -> stronger overfit /
# deploy-slow signal. Does the small-scale finding scale toward SDXL?
#   Q1 (primary): does sv=(s_slow+v_slow) still BEAT live=m_eff (deploy-slow win)?
#   Q2: does the 32-bit un-stranded combo (ratio+floor.1+gf_consol50) still MATCH the 64-bit gate?
# wd=0 (isolate implicit reg), shared init (seed0)+data_seed. Read: best val (live) + per-variant
# best deploy (live/sv/s2v/v/s) + --watch_accum (s_fast/s_slow/v_slow shares at scale).
set -u
cd /c/concord
BIG="--n_embd 640 --n_layer 10 --n_head 10 --block_size 256 --bsz 64"
BENCH="--data nanogpt_data/input.txt --max_iters 4000 --eval_interval 500 --eval_iters 50 --seed 0 --data_seed 1234 $BIG"
CONS="--eval_consolidated --watch_accum"
CONC="--mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --precond_p 0.5 --alpha 0.1 --step_cap 10 --rebalance_every 1 --concord_lr 5e-4 --concord_wd 0"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## big $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm bg_adamw   --mode adamw --adamw_lr 1e-3 --weight_decay 0          # overfit reference
run_arm bg_nogate  $CONC $CONS                                            # deploy-slow primary
run_arm bg_consol  $CONC $CONS --ratio_coh --ratio_chase_floor_min 0.1 \
                   --ratio_leak_floor_min 0.1 --gf_consol 50              # 32-bit un-stranded winner
run_arm bg_gate    $CONC $CONS --coh_gate                                 # 64-bit gate reference
echo "######## BIG DONE ########"
