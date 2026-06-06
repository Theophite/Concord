# Fused dequant-matmul — validation harnesses & experiment drivers

Dev scripts behind the CONCORD_FUSED_MATMUL work (OneTrainer-fork branch
`concord-integration`, impl in commit 16b77b91).

- fused_packed_test.py     correctness of the fused dequant-matmul (fwd y=x@Wt,
                           bwd grad_x=grad_y@W) vs materialize+cuBLAS + fp32 cross-check
- fused_packed_opt_test.py exp2-factored + coalesced-load variants; bit-identity + timing
- fused_autotune_test.py   @triton.autotune block sizes vs fixed 64^3; bit-identity + timing
- footprint_analyze.py     parse a Concord backup; packed-state breakdown by layer category
- phase2_sample.py         sample a Concord checkpoint in a fresh clean-memory process
- build_progression.py     grid montage of samples across epochs
- compare_ep45.py          montage comparing epoch 40/45/50
