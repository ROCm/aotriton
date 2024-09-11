#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os
import sys

from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    debug_fill_dropout_rng,
    AttentionExtraArgs
)
from _common_backward import _do_test_op_bwd
from _common_test import SdpaContextFromNPZ

@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('D_HEAD', [8, 63, 128])
@pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [1.2])
@pytest.mark.parametrize('storage_flip', [False])
def test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
@pytest.mark.parametrize('D_HEAD', [8, 203, 256])
@pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [1.2])
@pytest.mark.parametrize('storage_flip', [False])
def test_gqa(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('D_HEAD', [8, 203, 256])
@pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [1.2])
@pytest.mark.parametrize('storage_flip', [False])
def test_op_bwd_with_matrix_bias(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main_npz():
    SKIP_DK_DV = False
    SKIP_DQ = False
    SKIP_DB = True
    fn = sys.argv[1]
    ctx = SdpaContextFromNPZ(fn, dtype=torch.bfloat16, device='cuda')
    q, k, v, b = ctx.dev_tensors
    assert b is None, 'TODO: support bias in SdpaContextFromNPZ'
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)

    ext = AttentionExtraArgs(return_encoded_softmax=True,
                             autotune=False,
                             return_autotune=False)
    causal, sm_scale, dropout_p = ctx.sdpa_params[:3]
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    ctx.compute_ref_forward(ctx.sdpa_params)

    dout = ctx.dout
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff = ctx.validate_with_reference(tri_out, ctx.dout_tensors)
    assert is_allclose
    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    torch.set_printoptions(linewidth=200, threshold=4096)
    ctx.display_validation_results(tri_out, is_allclose, adiff, grads_allclose, grads_adiff)
    # Add more printing here
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    err_idx=(6, 2, 4, 0)
    print(f'{tri_dk[6, 2, 3, :]=}')
    print(f'{tri_dk[6, 2, 4, :]=}')
    print(f'{tri_dk[6, 2, 5, :]=}')
    print(f'{tri_dk[6, 1, 3, :]=}')
    print(f'{tri_dk[6, 1, 4, :]=}')
    print(f'{tri_dk[6, 1, 5, :]=}')
    print(f'{tri_dk[6, 3, 3, :]=}')
    print(f'{tri_dk[6, 3, 4, :]=}')
    print(f'{tri_dk[6, 3, 5, :]=}')

    print(f'{tri_dk[5, 1, 3, :]=}')
    print(f'{tri_dk[5, 1, 4, :]=}')
    print(f'{tri_dk[5, 1, 5, :]=}')
    print(f'{tri_dk[5, 3, 3, :]=}')
    print(f'{tri_dk[5, 3, 4, :]=}')
    print(f'{tri_dk[5, 3, 5, :]=}')

    print(f'{tri_dk[7, 1, 3, :]=}')
    print(f'{tri_dk[7, 1, 4, :]=}')
    print(f'{tri_dk[7, 1, 5, :]=}')
    print(f'{tri_dk[7, 3, 3, :]=}')
    print(f'{tri_dk[7, 3, 4, :]=}')
    print(f'{tri_dk[7, 3, 5, :]=}')
    print(f'{is_allclose=}')
    print(f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}')
    print(f'{adiff=} {grads_adiff=}')
    dk_nan = torch.argwhere(torch.isnan(tri_dk))
    def tdk(where):
        return (int(where[0]), int(where[1]), int(where[2]))
    leading_nan_idx = [tdk(where) for where in dk_nan]
    leading_nan_idx = sorted(list(set(leading_nan_idx)))
    leading_nan_idx = torch.tensor(leading_nan_idx, dtype=torch.int32)
    # leading_nan_idx = torch.tensor(set([tdk(where) for where in dk_nan]))
    print(f'{leading_nan_idx=}')

    import shutil
    with open('/proc/self/maps') as f:
        with open('maps.log', 'w') as o:
            shutil.copyfileobj(f, o)

def main2():
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-4-1
    # Memo: False-0.0-dtype0-0.0-False-4-256-8-1-4
    # False-1.2-dtype0-0.0-False-4-4-72-1-4
    BATCH = 8
    D_HEAD = 32
    N_HEADS = 8
    seqlen_q = 16
    seqlen_k = 16
    causal = False

    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main():
    BATCH = 1
    D_HEAD = 80
    N_HEADS = 2
    seqlen_q = 6432
    seqlen_k = 6432
    '''
    N_HEADS = 6432
    seqlen_q = 2
    seqlen_k = 2
    '''
    causal = False
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.bfloat16
    storage_flip = False
    bias_type = None
    _do_test_op_bwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    # main2()
    main_npz()
