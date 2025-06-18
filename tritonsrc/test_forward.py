#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os

from attn_torch_function import (
    attention,
    AttentionExtraArgs,
    debug_fill_dropout_rng,
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    CausalType,
)

FOR_RELEASE = bool(int(os.getenv('FOR_RELEASE', default='0')))

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    """
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    """
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    SPARSE_HEAD_SINCE = 5
    SPARSE_SEQ_SINCE = 5
    print(f'{query=}')
    print(f'{key=}')
    print(f'BEFORE softmax {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    # attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    print(f'BEFORE DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    if dropout_p > 0.0:
        if dropout_mask is not None:
            print(f'BEFORE DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            attn_weight.masked_fill_(dropout_mask.logical_not(), float("0.0"))
            print(f'AFTER DROPOUT_MASK {attn_weight[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            value = value / (1 - dropout_p)
        else:
            # assert False, "TESTING dropout_mask code path"
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    else:
        # assert False, "TESTING dropout_mask code path"
        pass
    print(f'{value[:,:, :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    av = attn_weight @ value
    print(f'{av[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
    return av, attn_weight

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

class FwdTester(object):

    def __init__(self):
        self.use_fill_rng_for_dropout = False

    def do_test_op_fwd(self, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if causal and bias_type is not None:
            pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
        torch.manual_seed(20)
        print(f"test_op_fwd {BATCH=}, {N_HEADS=}, {seqlen_q=}, {seqlen_k=}, {D_HEAD=}, {causal=}")
        SPARSE_HEAD_SINCE = 3
        SPARSE_SEQ_SINCE = 3
        Z = BATCH
        H = N_HEADS
        if True: # Real UT
            qdims = (BATCH, N_HEADS, seqlen_q, D_HEAD)
            kdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
            vdims = (BATCH, N_HEADS, seqlen_k, D_HEAD)
            bdims = (BATCH, N_HEADS, seqlen_q, seqlen_k)
            if storage_flip:
                qdims = (qdims[0], qdims[2], qdims[1], qdims[3])
                kdims = (kdims[0], kdims[2], kdims[1], kdims[3])
                vdims = (vdims[0], vdims[2], vdims[1], vdims[3])
                bdims = (bdims[0], bdims[2], bdims[1], bdims[3])
            q = (
                torch.empty(qdims, dtype=dtype, device="cuda")
                .normal_(mean=0., std=0.5)
                .requires_grad_()
            )
            k = (
                torch.empty(kdims, dtype=dtype, device="cuda")
                .normal_(mean=0., std=0.5)
                .requires_grad_()
            )
            v = (
                torch.empty(vdims, dtype=dtype, device="cuda")
                .normal_(mean=0., std=0.5)
                .requires_grad_()
            )
            if bias_type is None:
                b = None
            elif bias_type == 'matrix':
                b = torch.empty(bdims, dtype=dtype, device="cuda").normal_(mean=0., std=0.5)
            else:
                assert False, f'Unsupported bias_type {bias_type}'
            if storage_flip:
                q = torch.transpose(q, 1, 2)
                k = torch.transpose(k, 1, 2)
                v = torch.transpose(v, 1, 2)
                if b is not None:
                    b = torch.transpose(b, 1, 2)
                assert q.shape == (BATCH, N_HEADS, seqlen_q, D_HEAD)
                assert k.shape == (BATCH, N_HEADS, seqlen_k, D_HEAD)
                assert v.shape == (BATCH, N_HEADS, seqlen_k, D_HEAD)
        if False: # Debugging
            q = (
                torch.empty((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda")
                .normal_(mean=0., std=0.5)
                .requires_grad_()
            )
            k = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 1.0
            v = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 1.0
        if False:
            q = torch.ones((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda") * 1.0
            k = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 2.0
            v = torch.ones((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda") * 3.0
        if False:
            import numpy as np
            q = torch.arange(np.prod([Z, H, seqlen_q, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, seqlen_q, D_HEAD))
            k = torch.arange(np.prod([Z, H, seqlen_k, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, seqlen_q, D_HEAD))
            v = torch.arange(np.prod([Z, H, seqlen_k, D_HEAD]), dtype=dtype, device="cuda").reshape((Z, H, seqlen_q, D_HEAD))
            q = (q - 128.0) * 0.01
            k = (k - 128.0) * 0.01
            v = (v - 128.0) * 0.01
            q[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
            k[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
            v[:, :, :, SPARSE_HEAD_SINCE: ] = 0.0
            q[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
            k[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0
            v[:, :, SPARSE_SEQ_SINCE:, : ] = 0.0

        '''
        dout = torch.randn_like(q)
        # reference implementation
        M = torch.tril(torch.ones((N_CTX, N_CTX), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if causal:
            p[:, :, M == 0] = float("-inf")
        p = torch.softmax(p.float(), dim=-1).half()
        ref_out = torch.matmul(p, v)
        '''
        ext = AttentionExtraArgs(return_encoded_softmax=dropout_p > 0.0 and not self.use_fill_rng_for_dropout,
                                 autotune=False,
                                 return_autotune=False,
                                 fillnan=True)
        tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)

        if self.use_fill_rng_for_dropout:
            rdims = (BATCH, N_HEADS, seqlen_q, seqlen_k)
            if storage_flip:
                rdims = (rdims[0], rdims[2], rdims[1], rdims[3])
            r = torch.empty(rdims, device=q.device, dtype=torch.float32)
            if storage_flip:
                r = torch.transpose(r, 1, 2)
            philox_seed = DEFAULT_PHILOX_SEED
            philox_offset = DEFAULT_PHILOX_OFFSET
            debug_fill_dropout_rng(r, philox_seed, philox_offset)
            # Reserved for debugging use_fill_rng_for_dropout
            # print(f'{r[0,0,:16, :16]}=')
            # print(f'{r[0,0,:16, :16] > dropout_p}=')
            # print(f'{encoded_softmax[0,0,:16, :16] > 0}=')
            dropout_mask = r > dropout_p
        else:
            dropout_mask = encoded_softmax > 0 if encoded_softmax is not None else None
        # assert torch.allclose(dropout_mask, dropout_mask_naive)
        ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v,
                                                                    dropout_p=dropout_p,
                                                                    is_causal=causal,
                                                                    attn_mask=b,
                                                                    scale=sm_scale,
                                                                    dropout_mask=dropout_mask)
        if False:
            mref_out, mref_softmax = scaled_dot_product_attention(q, k, v,
                                         dropout_p=dropout_p,
                                         is_causal=causal,
                                         scale=sm_scale,
                                         dropout_mask=dropout_mask)
            print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{mref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{tri_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]/ref_out[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
            print(f'{q.shape=} {q.stride()=}')
            print(f'{k.shape=} {k.stride()=}')
            print(f'{v.shape=} {v.stride()=}')
            print(f'{encoded_softmax=}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.stride()=}')
                print(f'{encoded_softmax[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_SEQ_SINCE]=}')
                print(f'{dropout_mask.shape=} {dropout_mask.stride()=}')
                print(f'{dropout_mask[:,:,  :SPARSE_SEQ_SINCE, :SPARSE_HEAD_SINCE]=}')
        if dtype==torch.bfloat16:
            ATOL = 1e-1 * max(1.0, (seqlen_q + seqlen_k + D_HEAD) / 128.0)
        else:
            ATOL = 1e-2 * max(1.0, (seqlen_q + seqlen_k + D_HEAD) / 128.0)
        print(f'Using ATOL={ATOL}')
        is_allclose = torch.allclose(ref_out, tri_out, atol=ATOL, rtol=0)
        if not is_allclose:
            import numpy as np
            err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
            print(f'{err_idx=}')
            print(f'{tri_out[err_idx]=} {ref_out[err_idx]=} error: {tri_out[err_idx] - ref_out[err_idx]}')
        # if not is_allclose:
        if False:
            import numpy as np
            err_idx = np.unravel_index(torch.argmax(torch.abs(ref_out - tri_out)).cpu().numpy(), ref_out.shape)
            print(f'{tri_out[0][0][0][:]=}')
            print(f'{ref_out[0][0][0][:]=}')
            print(f'{mref_out[0][0][0][:]=}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax[0][0][0][:]=}')
            print(f'{ref_softmax[0][0][0][:]=}')
            print(f'{tri_out[-1][0][0][:]=}')
            print(f'{ref_out[-1][0][0][:]=}')
            print(f'{err_idx=}')
            print(f'{tri_out[err_idx]=}')
            print(f'{ref_out[err_idx]=}')
            if dropout_p > 0:
                # print(f'{unmasked_ref_out[0][0][0][:]=}')
                print(f'{dropout_mask[0][0][0][:]=}')
                print(f'{dropout_mask[err_idx]=}')
            # tri_cpu = tri_out[0, 0].cpu().detach().numpy()
            # print(f'{tri_cpu.shape=}')
        # compare
        assert is_allclose

def fmt_hdim(val):
    return f'hdim{val}'

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [8])
@pytest.mark.parametrize('D_HEAD', [8, 16, 21, 32, 64, 72, 96, 128, 160, 192, 203, 256], ids=fmt_hdim)
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2] if not FOR_RELEASE else [1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    tester = FwdTester()
    tester.do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 2, 4])
@pytest.mark.parametrize('N_HEADS', [1, 2, 4])
@pytest.mark.parametrize('D_HEAD', [16,32,64,128,256])
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [113])
# @pytest.mark.parametrize('seqlen_k', [79])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
def test_op_fwd_with_matrix_bias(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    tester = FwdTester()
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    tester.do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

@pytest.mark.parametrize('BATCH', [1, 4])
@pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
@pytest.mark.parametrize('causal', [False, True])
@pytest.mark.parametrize('storage_flip', [False, True])
def test_fill_dropout_rng(BATCH, N_HEADS, seqlen_q, seqlen_k, causal, storage_flip):
    D_HEAD = 128
    dropout_p = 0.5
    dtype = torch.float16
    sm_scale = 1.2
    bias_type = None
    tester = FwdTester()
    tester.use_fill_rng_for_dropout = True
    tester.do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

def main():
    BATCH = 1
    N_HEADS = 1
    D_HEAD = 16
    # seqlen_q = 512
    # seqlen_k = 4
    seqlen_q = 143
    seqlen_k = 256
    causal = True
    # seqlen_q = 16
    # seqlen_k = 32
    # causal = CausalType.BOTTOM_RIGHT
    sm_scale = 1.2
    dropout_p = 0.0
    dtype = torch.float16
    storage_flip = False
    bias_type = None
    tester = FwdTester()
    tester.do_test_op_fwd(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)

if __name__ == '__main__':
    main()
