# Fake empty-body kernel stub for ATI unit tests (triton-free).
# @ati.source AST-parses parameter NAMES only; bodies/annotations ignored.

def bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    cu_seqlens_q,
    num_seqlens,
    max_seqlen_q,
    hdim_vo,
    BLOCK_M,
    D_HEAD,
    PADDED_HEAD,
):
    pass

def bwd_preprocess_varlen(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    cu_seqlens_q,
    max_seqlen_q,
    seq_strides_q,
    hdim_vo,
    BLOCK_M,
    D_HEAD,
    PADDED_HEAD,
):
    pass
