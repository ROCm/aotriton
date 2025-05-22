#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os
import sys

from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    AttentionExtraArgs,
    BWD_FUSED,
    V3_API,
)
from _common_test import SdpaContext, SdpaParams, SdpaContextFromNPZ, AOTRITON_TORCH_ONLY_USE_CPU, fmt_hdim

@pytest.fixture()
def torch_gpu(worker_id):
    # Common worker_id values are "gw0", "gw1", etc.
    return int(worker_id[2:]) if worker_id != "master" else None

FOR_RELEASE = bool(int(os.getenv('FOR_RELEASE', default='0')))

POT_HEADDIMS = [16, 32, 64, 128, 256] + ([512] if not BWD_FUSED else [])
NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
# Prime head dimensions must be disabled
# PyTorch allocate tensors compactly by default. For example:
#   print(torch.rand((3,5,1033, 57), dtype=torch.float16, device='cuda').stride())
#   (294405, 58881, 57, 1)
# GPU kernels are unable to support unaligned memory access in any performant way
# PRIME_HEADDIMS = [7, 23, 37, 53, 67, 73, 83, 113, 149, 179, 211, 241] + ([401] if not BWD_FUSED else [])
# Multiple of 8 head dimensions are tested instead
M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216, 248] + ([408] if not BWD_FUSED else [])
PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919, 10601]
PRIME_SEQLEN_K = [13, 31, 41, 71, 223, 337, 571, 1063, 2081, 5237, 11369]

SMALL_HEADDIM_ONLY = bool(int(os.getenv('SMALL_HEADDIM_ONLY', default='0')))
LARGE_HEADDIM_ONLY = bool(int(os.getenv('LARGE_HEADDIM_ONLY', default='0')))

def remove_larger_than(data_list, threshold):
    return [x for x in data_list if x <= threshold]

def remove_not_larger_than(data_list, threshold):
    return [x for x in data_list if x > threshold]

def cdiv(x, div):
    return (x + div - 1) // div

def round_list_to_8x(data_list):
    return [cdiv(x, 8) * 8 for x in data_list]

if SMALL_HEADDIM_ONLY:
    POT_HEADDIMS = remove_larger_than(POT_HEADDIMS, 192)
    NPOT_HEADDIMS = remove_larger_than(NPOT_HEADDIMS, 192)
    # PRIME_HEADDIMS = remove_larger_than(PRIME_HEADDIMS, 192)
    M8_HEADDIMS = remove_larger_than(M8_HEADDIMS, 192)

if LARGE_HEADDIM_ONLY:
    POT_HEADDIMS = remove_not_larger_than(POT_HEADDIMS, 192)
    NPOT_HEADDIMS = remove_not_larger_than(NPOT_HEADDIMS, 192)
    # PRIME_HEADDIMS = remove_not_larger_than(PRIME_HEADDIMS, 192)
    M8_HEADDIMS = remove_not_larger_than(M8_HEADDIMS, 192)

REGULAR_HEADDIM_ONLY = bool(int(os.getenv('REGULAR_HEADDIM_ONLY', default='0')))
HEADDIM_8X_ONLY = bool(int(os.getenv('HEADDIM_8X_ONLY', default='0')))

assert not (REGULAR_HEADDIM_ONLY and HEADDIM_8X_ONLY), f'{REGULAR_HEADDIM_ONLY=} and {HEADDIM_8X_ONLY=} are mutually exclusive'

if REGULAR_HEADDIM_ONLY:
    ALL_HEADDIMS = POT_HEADDIMS + NPOT_HEADDIMS
elif HEADDIM_8X_ONLY:
    ALL_HEADDIMS = M8_HEADDIMS
else:
    ALL_HEADDIMS = POT_HEADDIMS + NPOT_HEADDIMS + M8_HEADDIMS

'''
Note: for now we cannot really test both fused and split kernel at the same
      time. Env var BWD_FUSED is used to make the switch.

      However we still add BWDOP to the tests arguments so we can easily tell
      the actual bwd op being tested.
'''
#TODO: Let BWDOP determine the real backward op at runtime

BWDOP_ids = ['Fused'] if BWD_FUSED else (['V3'] if V3_API else ['Split'])

def _make_block_eyes(q, base=1.0, inc=0.0):
    dhead = q.shape[-1]
    seqlen = q.shape[2]
    assert seqlen % dhead == 0
    scale = base
    for i in range(0, seqlen, dhead):
        q[:, :, i:i+dhead, :] = torch.eye(dhead, device=q.device, dtype=q.dtype) * scale
        scale += inc

def RP(x):
    rounded = 2 ** (x - 1).bit_length()
    return max(16, rounded)

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

def _do_test_op_bwd(args, device_str='cuda'):
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type = args
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    if BATCH > 1 and seqlen_q * seqlen_k >= 1024 * 1024:
        torch.cuda.empty_cache()
    SKIP_DK_DV = False
    SKIP_DQ = False
    SKIP_DB = True if bias_type is None else False
    USE_AUTOTUNE = True
    torch.manual_seed(20)
    transpose = (1, 2) if storage_flip else None
    ctx = SdpaContext(BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, dtype,
                      bias_type=bias_type, storage_flip=transpose, device=device_str, fillnan=True)
    ctx.create_ref_inputs()
    ctx.set_require_grads(skip_dq=SKIP_DQ, skip_dk_dv=SKIP_DK_DV, skip_db=SKIP_DB)
    q, k, v, b = ctx.dev_tensors
    # autotune = True
    # # triton implementation
    ext = AttentionExtraArgs(return_encoded_softmax=False if dropout_p == 0 else True,
                             autotune=False,
                             return_autotune=False,
                             fillnan=True)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    dropout_mask = encoded_softmax >= 0 if encoded_softmax is not None else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    dout = torch.rand_like(tri_out)
    ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff, tfts = ctx.validate_with_reference(tri_out, ctx.dout_tensors, return_target_fudge_factors=True)
    ctx.display_validation_results(tri_out, is_allclose, adiff, grads_allclose, grads_adiff)

    assert is_allclose, f'Forward pass {is_allclose=} {tfts=}'
    dq_allclose, dk_allclose, dv_allclose, db_allclose = grads_allclose
    tri_dq, tri_dk, tri_dv, tri_db = ctx.dout_tensors
    ref_dq, ref_dk, ref_dv, ref_db = ctx.dref_tensors
    if not SKIP_DQ:
        assert tri_dq is not None
        assert ref_dq is not None
    if not SKIP_DK_DV:
        assert tri_dk is not None
        assert tri_dv is not None
        assert ref_dk is not None
        assert ref_dv is not None
    if not SKIP_DB:
        assert tri_db is not None
        assert ref_db is not None
    assert dk_allclose and dv_allclose and dq_allclose and db_allclose, f'{dk_allclose=} {dv_allclose=} {dq_allclose=} {db_allclose=} {tfts=}'
    print(f'{tri_out=}')
    print(f'{adiff=} {grads_adiff=}')

def _test_op_bwd(args, device : int | None = None):
    if device is None:
        _do_test_op_bwd(args, device_str='cuda')
    else:
        with torch.cuda.device(device):
            _do_test_op_bwd(args, device_str=f'cuda:{device}')

# @pytest.mark.parametrize('BATCH', [1])
# @pytest.mark.parametrize('N_HEADS', [1])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [5])
# @pytest.mark.parametrize('D_HEAD', [16, 32, 64, 128, 256])
# Irregular-only PyTorch set
# @pytest.mark.parametrize('D_HEAD', [8, 21, 72, 96, 160, 192, 203])
# @pytest.mark.parametrize('seqlen_q', [1, 4, 32, 128, 256, 512, 1024, 7, 394, 250, 399, 511, 1019])
# @pytest.mark.parametrize('seqlen_k', [1, 4, 32, 128, 256, 512, 1024, 3, 217, 339, 313, 491, 988])
# PyTorch set
@pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 127, 256, 587, 1024, 2048])
# Minimal set
# @pytest.mark.parametrize('seqlen_q', [32, 128])
# @pytest.mark.parametrize('seqlen_k', [32, 128])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2] if not FOR_RELEASE else [1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_op_bwd(torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    _test_op_bwd(args, device=torch_gpu)

# @pytest.mark.parametrize('BATCH', [1, 4])
# @pytest.mark.parametrize('N_HEADS', [1, 4])
@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [3])
@pytest.mark.parametrize('N_HEADS', [1, 4] if not FOR_RELEASE else [5])
@pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
# @pytest.mark.parametrize('D_HEAD', [128])
# Complete set
# @pytest.mark.parametrize('seqlen_q', [4,8,16,17,32,64,128,143,256,512,1024,2048])
# @pytest.mark.parametrize('seqlen_k', [4,8,16,23,32,64,128,256,512,587,1024,2048])
# PyTorch set
@pytest.mark.parametrize('seqlen_q', [4, 8, 64, 143, 256, 512, 1024, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 8, 64, 128, 256, 587, 1024, 2048])
# @pytest.mark.parametrize('seqlen_q', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_k', [128,256,512,1024])
# @pytest.mark.parametrize('seqlen_q', [128, 113])
# @pytest.mark.parametrize('seqlen_k', [128, 79])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
# @pytest.mark.parametrize('dropout_p', [0.0])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
# @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('sm_scale', [0.0, 1.2] if not FOR_RELEASE else [1.2])
@pytest.mark.parametrize('storage_flip', [False, True])
# @pytest.mark.parametrize('return_encoded_softmax', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_op_bwd_with_matrix_bias(torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
    causal = False
    bias_type = 'matrix'
    '''
    _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
    '''
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    _test_op_bwd(args, device=torch_gpu)

@pytest.mark.parametrize('BATCH', [1, 4] if not FOR_RELEASE else [4])
@pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
# @pytest.mark.parametrize('D_HEAD', [8] if SMALL_HEADDIM_ONLY else [8, 203, 256])
@pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
@pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
@pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
@pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
@pytest.mark.parametrize('dropout_p', [0.0, 0.5])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize('sm_scale', [1.2])
@pytest.mark.parametrize('storage_flip', [False])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_gqa(torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
    bias_type = None
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    _test_op_bwd(args, device=torch_gpu)

if not FOR_RELEASE:  # Make the loading faster
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS)
    @pytest.mark.parametrize('seqlen_q', PRIME_SEQLEN_Q)
    @pytest.mark.parametrize('seqlen_k', PRIME_SEQLEN_K)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize('sm_scale', [1.2])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_irregulars(torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        _test_op_bwd(args, device=torch_gpu)

@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_large_bf16_nan_values(BWDOP):
    real_device = "cuda" if not AOTRITON_TORCH_ONLY_USE_CPU else "cpu"
    q = torch.full((1, 1, 1, 16), 133120.0, dtype=torch.bfloat16, device=real_device)
    k = torch.full((1, 1, 1, 16), 133120.0, dtype=torch.bfloat16, device=real_device)
    v = torch.full((1, 1, 1, 16), 133120.0, dtype=torch.bfloat16, device=real_device)
    b = None
    from torch.nn.functional import scaled_dot_product_attention
    from torch.nn.attention import sdpa_kernel, SDPBackend
    with sdpa_kernel(SDPBackend.MATH):
        out = scaled_dot_product_attention(q, k, v)
    print(out)

    causal = False
    sm_scale = 0.125
    dropout_p = 0
    ext = AttentionExtraArgs(return_encoded_softmax=causal,
                             autotune=False,
                             return_autotune=False)
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)

    print(tri_out)
    assert not torch.isnan(tri_out).any(), "Output should not contain NaNs!"

def main_npz():
    SKIP_DK_DV = False
    SKIP_DQ = False
    SKIP_DB = True
    fn = sys.argv[1]
    ctx = SdpaContextFromNPZ(fn, dtype=None, device='cuda')
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
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    _test_op_bwd(args)

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
    args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
    _test_op_bwd(args)

if __name__ == '__main__':
    main2()
    # main_npz()
