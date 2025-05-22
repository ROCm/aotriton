#!/usr/bin/env python
# Copyright © 2024-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import pytest
import torch

from _common_test import SdpaContext, SdpaParams
from attn_torch_function import attention, AttentionExtraArgs, PersistentType

SKIP_DK_DV = bool(int(os.getenv('SKIP_DK_DV', default='0')))
SKIP_DQ = bool(int(os.getenv('SKIP_DQ', default='0')))
SKIP_DB = bool(int(os.getenv('SKIP_DB', default='0')))

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

def _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    skip_dk_dv = SKIP_DK_DV
    skip_dq = SKIP_DQ
    skip_db = True if bias_type is None else SKIP_DB
    torch.manual_seed(20)
    SPARSE_HEAD_SINCE = 1
    SPARSE_SEQ_SINCE = 1
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device='cuda')
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=skip_dq, skip_dk_dv=skip_dk_dv, skip_db=skip_db)

    q, k, v, b = ctx.dev_tensors
    # autotune = True
    # # triton implementation
    ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0,
                             autotune=False,
                             return_autotune=False,
                             fillnan=True,
                             persistent_type=PersistentType.AUTOSELECT)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    dropout_mask = encoded_softmax >= 0 if dropout_p > 0.0 else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)
    dout = torch.rand_like(tri_out)
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff, tfts = ctx.validate_with_reference(tri_out, ctx.dout_tensors, return_target_fudge_factors=True)
    if not is_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=}')
        print(f'{ref_out[err_idx]=}')
        print(f'{tri_out[0, 0, :4, :4]=}')
        print(f'{ref_out[0, 0, :4, :4]=}')
        print(f'{tri_out[0, 0, 0, 64:]=}')
    assert is_allclose, 'Forward pass {is_allclose=}'

    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=dtype)
    if not dv_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{q[0,0,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[0,0,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[0,0,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        if dropout_mask is not None:
            print(f'{dropout_mask.shape=}')
            print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{err_idx=}')
        print(f'{tri_dv[err_idx]=}')
        print(f'{ref_dv[err_idx]=}')
        print(f'{torch.isnan(ref_dv).any()=}')
        '''
        any_nan = torch.isnan(ref_dv).any()
        if any_nan:
            torch.set_printoptions(linewidth=200)
            print(f'{q=}')
            print(f'{k=}')
            print(f'{v=}')
            print(f'{dropout_p=}')
            print(f'{causal=}')
            print(f'{sm_scale=}')
        '''
        if seqlen_q <= 16 or True:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dk[0,0, :4, :4]=}')
            print(f'{ref_dk[0,0, :4, :4]=}')
            print(f'{tri_dv[0,0, :4, :4]=}')
            print(f'{ref_dv[0,0, :4, :4]=}')
            print(f'{tri_dv[0,0]=}')
            print(f'{ref_dv[0,0]=}')

    if dv_allclose and not dk_allclose:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        # print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        if dropout_mask is not None:
            print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        if seqlen_q <= 16 or True:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dk[0,0]=}')
            print(f'{ref_dk[0,0]=}')

    if dk_allclose and dv_allclose and not dq_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')
        if seqlen_q <= 16 or True:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dq[0,0, :4, :4]=}')
            print(f'{ref_dq[0,0, :4, :4]=}')

    if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
        import numpy as np
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
        print(f'{err_idx=}')
        print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')
    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=} {tfts=}'
    print(f'{adiff=} {grads_adiff=}')

