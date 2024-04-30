#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import numpy as np
import torch

from attn_torch_function import attention
from _common_test import SdpaContextFromDump_InputOnly, SdpaContextFromDump_Complete, SdpaParams

SPARSE_SEQ_SINCE = 4
SPARSE_HEAD_SINCE = 4

def main():
    d = np.load('dump_attn_mask.npz')
    ctx = SdpaContextFromDump_InputOnly(d['q_ref_lp'], d['k_ref_lp'], d['v_ref_lp'], d['b_ref_lp'])
    ctx.create_ref_inputs()
    ctx.set_require_grads()

    q, k, v, b = ctx.dev_tensors
    causal = bool(d['is_causal'])
    assert d['scale'] == 0
    assert q.shape[-1] == 16
    sm_scale = 0.25  # scale comes from AMD_LOG_LEVEL=3
    dropout_p = float(d['dropout_p'])
    return_encoded_softmax = True
    USE_AUTOTUNE = True

    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, return_encoded_softmax, USE_AUTOTUNE)
    dropout_mask = encoded_softmax >= 0
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)
    dout, = ctx._load([d['upstream_grad']])
    ctx.compute_backward(tri_out, dout)
    is_allclose, grads_allclose = ctx.validate_with_reference(tri_out, ctx.dout_tensors)

    if not is_allclose:
        ref_out = ctx.lp_refout_tensors[0]
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=} {tri_out.dtype=}')
        print(f'{ref_out[err_idx]=}')
        print(f'{d["out_ref"][err_idx]=}')
        print(f'{d["out_lp_ref"][err_idx]=}')
    assert is_allclose, 'Forward pass {is_allclose=}'

    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    print(f'{ctx.dout_tensors=}')
    print(f'{ctx.dref_tensors=}')
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=q.dtype)
    if not dv_allclose:
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{dropout_mask.shape=}')
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
        if seqlen_q <= 16:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dk[0,0]=}')
            print(f'{ref_dk[0,0]=}')
            print(f'{tri_dv[0,0]=}')
            print(f'{ref_dv[0,0]=}')
            # print(f'{tri_dq[0,0]=}')
            # print(f'{ref_dq[0,0]=}')

    if dv_allclose and not dk_allclose:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

    if dk_allclose and dv_allclose and not dq_allclose:
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

    if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
        print(f'{err_idx=}')
        print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')

    if k.shape[2] == 1024:
        max_aindex = (0, 2, 915, 50)
        max_rindex = (0, 1, 663, 11)
    elif k.shape[2] == 128:
        max_aindex = (0, 2, 93, 11)
        max_rindex = (0, 2, 123, 3)
    else:
        assert False
    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}'
    print(f'-----------data from dump--------------')
    dk = d['dk']
    dk_ref = d['dk_ref']
    dk_ref_lp = d['dk_ref_lp']
    b_ref = d['b_ref']
    print(f'{b_ref=}')
    print(f'{dk[max_aindex]=}')
    print(f'{dk[max_aindex]=}')
    print(f'{dk_ref[max_aindex]=}')
    print(f'{dk_ref_lp[max_aindex]=}')
    print(f'-----------data from compute--------------')
    b_ref = ctx.ref_tensors[3]
    dq, dk, dv, db = ctx.dout_tensors
    dk_ref = ctx.dref_tensors[1]
    dk_ref_lp = ctx.lp_dref_tensors[1]
    print(f'{b_ref=}')
    print(f'{dk_ref[max_aindex]=}')
    print(f'{dk_ref_lp[max_aindex]=}')
    print(f'{dk[max_aindex]=}')
    print(f'{dk[max_aindex]=}')
    print(f'------------------------------------------')

if __name__ == '__main__':
    main()


def main2():
    d = np.load('dump_attn_mask.npz')
    ctx = SdpaContextFromDump_Complete(d['q_ref_lp'], d['k_ref_lp'], d['v_ref_lp'], d['b_ref_lp'])
    ctx.set_require_grads()
    q, k, v, b = ctx.dev_tensors
    assert q.requires_grad
    assert k.requires_grad
    assert v.requires_grad
    assert b.requires_grad
    # preprocess_mask() in aten/src/ATen/native/transformers/attention.cpp
    b = b.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    causal = bool(d['is_causal'])
    sm_scale = float(d['scale'])
    dropout_p = float(d['dropout_p'])
    return_encoded_softmax = False
    USE_AUTOTUNE = False
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, return_encoded_softmax, USE_AUTOTUNE)
    ctx.compute_ref_forward(d['out_ref'], d['out_lp_ref'])
    D_ref = (d['dq_ref'], d['dk_ref'], d['dv_ref'], d['db_ref'])
    D_ref_lp = (d['dq_ref_lp'], d['dk_ref_lp'], d['dv_ref_lp'], d['db_ref_lp'])
    ctx.compute_backward(tri_out, d['upstream_grad'], D_ref, D_ref_lp)
    # assert q.grad is not None
    is_allclose, grads_allclose = ctx.validate_with_reference(tri_out, ctx.dout_tensors)

    if not is_allclose:
        ref_out = ctx.lp_refout_tensors[0]
        err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
        print(f'{err_idx=}')
        print(f'{tri_out[err_idx]=} {tri_out.dtype=}')
        print(f'{ref_out[err_idx]=}')
        print(f'{d["out_ref"][err_idx]=}')
        print(f'{d["out_lp_ref"][err_idx]=}')
    assert is_allclose, 'Forward pass {is_allclose=}'

    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    def TO(ref_tensor):
        return ref_tensor.to(device=q.device, dtype=dtype)
    if not dv_allclose:
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dv) - tri_dv)).cpu().numpy(), ref_dv.shape)
        print(f'{q.shape=} {q.stride()=} {q.dtype=}')
        print(f'{k.shape=} {k.stride()=} {k.dtype=}')
        print(f'{v.shape=} {v.stride()=} {v.dtype=}')
        print(f'{q[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{k[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{v[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        # print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{dropout_mask.shape=}')
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
        if seqlen_q <= 16:
            torch.set_printoptions(linewidth=200, threshold=4096)
            print(f'{tri_dk[0,0]=}')
            print(f'{ref_dk[0,0]=}')
            print(f'{tri_dv[0,0]=}')
            print(f'{ref_dv[0,0]=}')
            # print(f'{tri_dq[0,0]=}')
            # print(f'{ref_dq[0,0]=}')

    if dv_allclose and not dk_allclose:
        print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        print(f'{ref_dk[:,:,  :SPARSE_SEQ_SINCE+1, :SPARSE_HEAD_SINCE+1]=}')
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dk) - tri_dk)).cpu().numpy(), ref_dk.shape)
        print(f'{err_idx=}')
        print(f'{tri_dk[err_idx]=} {ref_dk[err_idx]=} error = {torch.abs(tri_dk[err_idx] - ref_dk[err_idx])}')
        print(f'{tri_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_dk[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')

    if dk_allclose and dv_allclose and not dq_allclose:
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_dq) - tri_dq)).cpu().numpy(), ref_dq.shape)
        print(f'{err_idx=}')
        print(f'{tri_dq[err_idx]=} {ref_dq[err_idx]=} error = {torch.abs(tri_dq[err_idx] - ref_dq[err_idx])}')

    if dk_allclose and dv_allclose and dq_allclose and not db_allclose:
        err_idx = np.unravel_index(torch.argmax(torch.abs(TO(ref_db) - tri_db)).cpu().numpy(), ref_db.shape)
        print(f'{err_idx=}')
        print(f'{tri_db[err_idx]=} {ref_db[err_idx]=} error = {torch.abs(tri_db[err_idx] - ref_db[err_idx])}')
    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=}'

