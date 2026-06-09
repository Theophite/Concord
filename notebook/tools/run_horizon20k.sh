#!/usr/bin/env bash
# Push to 20k: does the Concord-AdamW gap keep ~halving per horizon-doubling
# (geometric close to ~0) or settle at a persistent int-packing floor?
#   5k gap 0.0765, 10k gap 0.0373.  Both 20k, cosine spanning 20k, same init +
#   batch order (seed 0, data_seed 1234). 20k*64*256 ~= 3.3 passes over enwik8.
# Report train AND val: at 10k AdamW began to overfit (val bounce 1.0226->1.0295);
# watch whether 20k AdamW overfits more (train<<val) while Concord (noise-resistant)
# keeps descending -- that would close/cross the gap for a DIFFERENT reason.
set -u
cd /c/concord
BENCH="--data nanogpt_data/enwik8 --max_iters 20000 --eval_interval 1000 --eval_iters 50 --seed 0 --data_seed 1234"
run_arm() {
  local tag="$1"; shift
  for a in 1 2 3; do
    echo "######## h20k $tag (attempt $a) ########"
    [ "$a" -gt 1 ] && { taskkill //F //IM python.exe 2>/dev/null; sleep 25; }
    python src/train_nanogpt.py $BENCH "$@" --tag "$tag" && return 0
    echo "[$tag] retry"; sleep 15
  done
}
run_arm h20k_concord --mode concord --v_scale 0 --gf_trust_delta_sq 1 --eps 1e-10 --concord_lr 5e-4 --coh_gate --eval_consolidated
run_arm h20k_adamw   --mode adamw --adamw_lr 1e-3 --weight_decay 0.1
echo "######## H20K DONE ########"
