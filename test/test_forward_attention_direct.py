
import torch
import os
import math
import time

from attn_torch_function import (
    attention,
    AttentionExtraArgs
)
from _common_test import SdpaContext

def _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if causal and bias_type is not None:
        print("Skipping: _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
        return
    if sm_scale == 'l1':
        sm_scale = 1.0 / D_HEAD
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(D_HEAD)
    
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device='cuda')
    ctx.create_ref_inputs()
    q, k, v, b = ctx.dev_tensors

    ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                             autotune=False,
                             return_autotune=False)
    print("Warming up")
    attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    torch.cuda.synchronize()
    print(f"Benchmarking (B={BATCH}, H={N_HEADS}, D={D_HEAD}, Sc={sm_scale})")

    with torch.no_grad():
        for i in range(20):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            print("time", end_time - start_time)

def main():
    # set arg
    BATCH = 1
    N_HEADS = 24
    D_HEAD = 64
    seqlen_q = 4250
    seqlen_k = 4250
    causal = False
    sm_scale = 0.0
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    main()