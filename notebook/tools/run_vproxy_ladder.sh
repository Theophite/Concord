#!/usr/bin/env bash
# v_proxy ENGAGEMENT ladder on enwik8. The eps=1e-6/lr=1e-3 arm was stable but
# eps(1e-6) >> v_proxy median(~5e-9) -> denom dominated by eps -> UNIFORM rescale,
# not per-coord whitening (hence it parked at ~1.3955, next to inert 1.4436).
# Real whitening needs eps < v_proxy. Walk eps DOWN through the v_proxy scale;
# above the median hold the eps-dominated step-scale fixed (lr ~ sqrt(eps));
# one WARM-lr deep-eps arm = full per-coord whitening leaning on step_cap=10.
# 2500-iter SCAN (fast); same seed/data_seed as the comparability run so vals
# are directly comparable to inert-Concord. Auto-waits for the comparability
# run to free the GPU. Per-arm retry (unstable box).
set -u
cd /c/concord
OUT=C:/concord/compare_out
LADLOG="$OUT/vproxy_ladder.txt"
mkdir -p "$OUT"
COMMON="--data nanogpt_data/enwik8 --max_iters 2500 --eval_interval 250 \
  --eval_iters 50 --seed 0 --data_seed 1234 --precond_p 0.5"

# --- wait for the comparability run to release the GPU (its last artifact) ---
echo "[ladder] waiting for comparability run to finish (e8gap_adamw.pt)..."
for w in $(seq 1 120); do
  [ -f "$OUT/e8gap_adamw.pt" ] && { echo "[ladder] GPU free, starting."; break; }
  sleep 20
done

run_arm () {
  local eps="$1"; local lr="$2"; local tag="$3"
  for attempt in 1 2 3; do
    echo "######## ladder $tag : eps=$eps concord_lr=$lr (attempt $attempt) ########"
    [ "$attempt" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py --mode concord $COMMON \
      --eps "$eps" --concord_lr "$lr" --tag "$tag" && return 0
    echo "[$tag] crashed; retrying"; sleep 15
  done
  echo "[$tag] FAILED"; return 1
}

echo "==== v_proxy ladder (val@2500; inert-Concord ref ~1.50 @2500) ====" > "$LADLOG"
# eps walked down through v_proxy median ~5e-9; lr ~ sqrt(eps) above median.
run_arm 1e-7  3.2e-4 vp_e7        # control: eps-dominated, expect ~inert
run_arm 1e-8  1e-4   vp_e8        # near transition
run_arm 1e-9  3.2e-5 vp_e9        # below median: whitening engages (diagonal, cool)
run_arm 1e-10 1e-5   vp_e10       # deep: diagonal, cold
run_arm 1e-10 1e-4   vp_e10_warm  # deep eps + WARM lr: full whitening on step_cap
echo "######## VPROXY LADDER DONE ########"
