#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os
import math
import time

from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    AttentionExtraArgs
)
from _common_test import SdpaContext, SdpaParams

FOR_RELEASE = bool(int(os.getenv('FOR_RELEASE', default='0')))

'''
Flash Attention is batch operator that evaluates sm(QK')V
Q = batch_size x ... x seqlen_q x head_size
K = batch_size x ... x seqlen_k x head_size
    => K' = batch_size x ... x head_size x seqlen_k
V = batch_size x ... x seqlen_k x head_size
sm(.) = softmax(.)
The output size is
batch_size x ... x seqlen_q x head_size

Note: In Flash V2 API the ... is denoted as "num_heads", serving as uniformly sized sequences
but in PyTorch API it does not present at all
'''

def _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    if sm_scale == 'l1':
        sm_scale = 1.0 / D_HEAD
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(D_HEAD)
    # if BATCH > 1 and seqlen_q >= 1024 and seqlen_k >= 1024:
    #     torch.cuda.empty_cache()
    SKIP_DK_DV = True
    SKIP_DQ = True
    SKIP_DB = True if bias_type is None else False
    USE_AUTOTUNE = False
    torch.manual_seed(20)
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)
    q, k, v, b = ctx.dev_tensors
    # triton implementation
    ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                             autotune=False,
                             return_autotune=False)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    print("warnup end")

    t_time = []
    for i in range(20):
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        print("time",end_time - start_time)



@pytest.mark.parametrize('BATCH', [1] )
@pytest.mark.parametrize('N_HEADS', [24])
@pytest.mark.parametrize('D_HEAD', [64] )
@pytest.mark.parametrize('seqlen_q,seqlen_k', [
    # (1024, 1024),
    # (1178, 1178),
    (4250, 4250)
    # (4096, 4096)
], ids=['4250'])
@pytest.mark.parametrize('causal', [False], ids=['CausalOff'])
@pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16])
@pytest.mark.parametrize('sm_scale', [0.0])
@pytest.mark.parametrize('storage_flip', [False])
def test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)


dtype0 = torch.float16
dtype1 = torch.bfloat16
dtype2 = torch.float32

# Testing test_op_fwd_with_matrix_bias from string
def main4():
    utshort = 'False-1.2-dtype0-0.0-4-2048-32-1-1'
    utlist_str = list(reversed(utshort.split('-')))
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dropout_p, dtype, sm_scale, storage_flip = [eval(e) for e in utlist_str]
    causal = False
    bias_type = 'matrix'
    _do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    main4()
