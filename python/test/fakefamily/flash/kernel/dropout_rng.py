# Fake empty-body kernel stub for ATI unit tests (triton-free).
# @ati.source AST-parses parameter NAMES only; bodies/annotations ignored.

def debug_fill_dropout_rng(
    R,
    stride_rz,
    stride_rh,
    stride_rm,
    stride_rn,
    seqlen_q,
    seqlen_k,
    philox_seed,
    philox_offset_base,
    BLOCK_M,
    BLOCK_N,
):
    pass

def debug_fill_dropout_rng_tensor(
    R,
    stride_rz,
    stride_rh,
    stride_rm,
    stride_rn,
    seqlen_q,
    seqlen_k,
    philox_seed_ptr,
    philox_offset_base_ptr,
    BLOCK_M,
    BLOCK_N,
):
    pass

def debug_simulate_encoded_softmax(
    R,
    stride_rz,
    stride_rh,
    stride_rm,
    stride_rn,
    dropout_p,
    Num_head_q,
    Max_seqlen_q,
    Max_seqlen_k,
    philox_seed_ptr,
    philox_offset1,
    philox_offset2,
    BLOCK_M,
    BLOCK_N,
):
    pass
