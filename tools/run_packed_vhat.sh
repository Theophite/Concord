#!/usr/bin/env bash
# PACKED-INT validation of the rank-1 v-hat result. Concord packed kernel with
# v_scale=0 (kill velocity-noise v_proxy) + gf_trust_delta_sq=1 -> denom =
# (v_hat+eps)^0.5, v_hat = Adafactor rank-1 E[g^2] (fp32 v_row/v_col, out+in
# floats/layer). The int-packed analog of FactoredAdam. Same init + batch order
# (seed 0, data_seed 1234). TARGET: land near Adam 1.07, NOT SGD-chase 1.43.
set -u
cd /c/concord
OUT=C:/concord/compare_out
COMMON="--mode concord --data nanogpt_data/enwik8 --max_iters 5000 \
  --eval_interval 250 --eval_iters 50 --seed 0 --data_seed 1234 \
  --v_scale 0 --gf_trust_delta_sq 1 --precond_p 0.5 --alpha 0.1 --step_cap 10"

run_arm() {
  local eps="$1" lr="$2" tag="$3" save="$4"
  for a in 1 2 3; do
    if [ -n "$save" ] && [ -f "${save}_concord.pt" ]; then echo "[$tag] done"; return 0; fi
    echo "######## packed-vhat $tag eps=$eps lr=$lr (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    if [ -n "$save" ]; then
      python src/train_nanogpt.py $COMMON --eps "$eps" --concord_lr "$lr" \
        --tag "$tag" --save_prefix "$save"
      [ -f "${save}_concord.pt" ] && return 0
    else
      python src/train_nanogpt.py $COMMON --eps "$eps" --concord_lr "$lr" \
        --tag "$tag" && return 0
    fi
    echo "[$tag] retry"; sleep 15
  done
}

run_arm 1e-10 1e-3 pv_e10_l1e3 "$OUT/e8pvhat"   # primary (matched scale), saved
run_arm 1e-10 3e-3 pv_e10_l3e3 ""               # hotter lr
run_arm 1e-8  1e-3 pv_e8_l1e3  ""               # higher eps floor
echo "######## PACKED VHAT RUNS DONE ########"
