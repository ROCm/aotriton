#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import pytest
import torch
import os
import sys
import math
import pathlib
import fcntl
import struct
import itertools
import json
import gc

from attn_torch_function import (
    DEFAULT_PHILOX_SEED,
    DEFAULT_PHILOX_OFFSET,
    attention,
    AttentionExtraArgs,
    BWD_IMPL,
    V3_API,
    PROBE_UNSUPPORTED,
    hipError_t,
    hipGetLastError,
)
from _common_test import (
    SdpaContext,
    SdpaParams,
    SdpaContextFromNPZ,
    AOTRITON_TORCH_ONLY_USE_CPU,
    fmt_hdim,
    fmt_nheads,
)

ON_GPU = os.getenv('ON_GPU', default=None)
RECORD_ADIFFS_TO = os.getenv('RECORD_ADIFFS_TO', default=None)
USE_ADIFFS_TXT = os.getenv('USE_ADIFFS_TXT', default=None)

if USE_ADIFFS_TXT is not None:
    adiffs = {}
    with open(USE_ADIFFS_TXT) as f:
        for line in f:
            utname, adiff_str = line.rstrip().split('\t')
            if adiff_str == "OOM":
                adiffs[utname] = "OOM"
            elif adiff_str == "NAN":
                adiffs[utname] = "NAN"
            else:
                adiffs[utname] = json.loads(adiff_str)
else:
    adiffs = {}

# SIGSEGV_ERROR_CODE = signal.SIGSEGV

def exit_pytest():
    # os.kill(os.getpid(), SIGSEGV_ERROR_CODE)
    os._exit(139)

PYTEST_XDIST_WORKER_COUNT=int(os.getenv('PYTEST_XDIST_WORKER_COUNT', default='0'))
STRUCT_FLOCK = 'hhllh'
PAGE_SIZE = 4096

if PYTEST_XDIST_WORKER_COUNT == 0:  # No pytest
    @pytest.fixture()
    def torch_gpu():
        return 0
elif ON_GPU is not None:
    @pytest.fixture()
    def torch_gpu():
        yield int(ON_GPU)
        return
else:
    @pytest.fixture(scope="session", autouse=True)
    def gpufilelock(tmp_path_factory, testrun_uid):
        # get the temp directory shared by all workers
        root_tmp_dir = tmp_path_factory.getbasetemp().parent
        lockfile = root_tmp_dir / "gpulock"
        with open(lockfile, 'wb') as f:
            f.seek(PYTEST_XDIST_WORKER_COUNT * PAGE_SIZE - 1)
            f.write(b'\0')
        return lockfile

    @pytest.fixture(scope="session")  # For pytest-xdist, "session" scope is per-worker process
    def torch_gpu(worker_id, testrun_uid, gpufilelock):
        with open(gpufilelock, 'wb') as f:
            for gpu in itertools.cycle(range(PYTEST_XDIST_WORKER_COUNT)):
                ld = struct.pack(STRUCT_FLOCK, fcntl.F_WRLCK, os.SEEK_SET, PAGE_SIZE * gpu, PAGE_SIZE, 0)
                try:
                    ret = fcntl.fcntl(f, fcntl.F_SETLK, ld)
                    print(f'{worker_id} uses GPU {gpu} filelock = {gpufilelock}', file=sys.stderr, flush=True)
                    yield gpu
                    ud = struct.pack(STRUCT_FLOCK, fcntl.F_UNLCK, os.SEEK_SET, PAGE_SIZE * gpu, PAGE_SIZE, 0)
                    ret = fcntl.fcntl(f, fcntl.F_SETLK, ud)
                    return
                except BlockingIOError as e:
                    pass

FOR_RELEASE = int(os.getenv('FOR_RELEASE', default='0'))
SMALL_VRAM = bool(int(os.getenv('SMALL_VRAM', default='0')))

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

if BWD_IMPL == 0:
    POT_HEADDIMS = [16, 32, 64, 128, 256, 512]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216, 248, 408]
elif BWD_IMPL == 1:
    POT_HEADDIMS = [16, 32, 64, 128, 256]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192, 224]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184, 216]
elif BWD_IMPL == 2:
    POT_HEADDIMS = [16, 32, 64, 128]
    NPOT_HEADDIMS = [48, 80, 96, 160, 192]
    M8_HEADDIMS = [8, 24, 40, 56, 72, 88, 96, 120, 152, 184]
    DTYPES = [torch.float16, torch.bfloat16]
else:
    assert False, f'Unsupported BWD_IMPL {BWD_IMPL}'
# Prime head dimensions must be disabled
# PyTorch allocate tensors compactly by default. For example:
#   print(torch.rand((3,5,1033, 57), dtype=torch.float16, device='cuda').stride())
#   (294405, 58881, 57, 1)
# GPU kernels are unable to support unaligned memory access in any performant way
# PRIME_HEADDIMS = [7, 23, 37, 53, 67, 73, 83, 113, 149, 179, 211, 241] + ([401] if not BWD_IMPL else [])
# Multiple of 8 head dimensions are tested instead
REGULAR_SEQLEN = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
REGULAR_SEQLEN_2K = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]  # OOM when test with bias
PRIME_SEQLEN_Q = [11, 17, 37, 67, 157, 257, 523, 1033, 2063, 4919]
PRIME_SEQLEN_K = [13, 31, 41, 71, 223, 337, 571, 1063, 2081, 5237]

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

ALL_HEADDIMS = POT_HEADDIMS + NPOT_HEADDIMS + M8_HEADDIMS

# Deduplication
ALL_HEADDIMS = sorted(list(set(ALL_HEADDIMS)))

'''
Note: for now we cannot really test both fused and split kernel at the same
      time. Env var BWD_IMPL is used to make the switch.

      However we still add BWDOP to the tests arguments so we can easily tell
      the actual bwd op being tested.
'''
#TODO: Let BWDOP determine the real backward op at runtime

def _get_BWDOP_id():
    if V3_API:
        return 'V3'
    if BWD_IMPL == 2:
        return 'AITERASM'
    if BWD_IMPL == 1:
        return 'Fused'
    if BWD_IMPL == 0:
        return 'Split'
    assert False, f'Unsupported BWD_IMPL {BWD_IMPL}'

BWDOP_ids = [_get_BWDOP_id()]

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

def _do_test_op_bwd(request, args, device_str='cuda'):
    BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type = args
    if sm_scale == 'l1':
        sm_scale = 1.0 / D_HEAD
    elif sm_scale == 'l2':
        sm_scale = 1.0 / math.sqrt(D_HEAD)
    if BWD_IMPL == 2:  # AITER ASM
        if dropout_p > 0.0:
            pytest.skip("Dropout unsupported in AITER ASM backend for now. Need adjust FWD PRNG function")
        if D_HEAD < 64:
            pytest.skip("hdim < 64 AITER ASM kernel does not exist.")
        if D_HEAD > 192:
            pytest.skip("hdim > 192 AITER ASM kernel does not exist.")
    if causal and bias_type is not None:
        pytest.skip("_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True")
    if SMALL_VRAM and seqlen_q * seqlen_k * D_HEAD > 4096 * 8192 * 256:
        pytest.skip("Skip large tests (qkd > 4096 * 8192 * 256) due to low VRAM.")
    if 'gfx11' in torch.cuda.get_device_properties(0).gcnArchName:
        if D_HEAD > 256:
            pytest.skip("Skip hdim > 256 on gfx11 arch due to register pressure.")
    utname = os.environ.get('PYTEST_CURRENT_TEST')
    use_adiff_entry = adiffs.get(utname, None)
    if use_adiff_entry == "OOM":
        pytest.skip("[Adiffs] Skip due to known OOM.")
    if use_adiff_entry == "NAN":
        mark = pytest.mark.xfail(reason="[Adiffs] XPASS due to known NAN.")
        request.node.add_marker(mark)
        return 0
    print(f"{use_adiff_entry=}")
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
                             fillnan=True,
                             illaddr_handler=exit_pytest,
                             )
    tri_out, encoded_softmax, _ = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    dropout_mask = encoded_softmax >= 0 if encoded_softmax is not None else None
    sdpa_params = SdpaParams(causal=causal, sm_scale=sm_scale, dropout_p=dropout_p, dropout_mask=dropout_mask)
    ref_out, _ = ctx.compute_ref_forward(sdpa_params)

    dout = torch.rand_like(tri_out)
    if PROBE_UNSUPPORTED:
        try:
            ctx.compute_backward(tri_out, dout)
        except NotImplementedError as e:
            pytest.xfail("Unsupported Config in AITER")
    else:
        ctx.compute_backward(tri_out, dout)
    is_allclose, adiff, grads_allclose, grads_adiff, tfts = ctx.validate_with_reference(tri_out, ctx.dout_tensors, return_target_fudge_factors=True, use_adiff_entry=use_adiff_entry)
    ctx.display_validation_results(tri_out, is_allclose, adiff, grads_allclose, grads_adiff)

    if RECORD_ADIFFS_TO is not None and (not is_allclose or not all(grads_allclose)):
        with open(RECORD_ADIFFS_TO, 'a') as f:
            dj = { "adiff" : adiff, "grads_adiff" : grads_adiff }
            print(utname, "\t", json.dumps(dj), file=f, flush=True, sep='')
        pytest.xfail(f"RECORD ADIFFS {adiff=} {grads_adiff=}")
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
    return seqlen_q * seqlen_k * D_HEAD

def _test_op_bwd(request, args, device : int | None = None):
    try:
        if device is None:
            qkh = _do_test_op_bwd(request, args, device_str='cuda')
        else:
            with torch.cuda.device(device):
                qkh = _do_test_op_bwd(request, args, device_str=f'cuda:{device}')
        if qkh > 2048 * 2048 * 128:
            gc.collect()
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if hipGetLastError() == hipError_t.hipErrorIllegalAddress:
            exit_pytest()
        raise e

if FOR_RELEASE == 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5, (10, 2)], ids=fmt_nheads)
    @pytest.mark.parametrize('D_HEAD', [8, 64, 184], ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [11, 523, 2048])
    @pytest.mark.parametrize('seqlen_k', [31, 337, 1063])
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 2 else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_fast(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        _test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', REGULAR_SEQLEN)
    @pytest.mark.parametrize('seqlen_k', REGULAR_SEQLEN)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 2 else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_regular_bwd(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        _test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 0 and BWD_IMPL != 2:  # AITER ASM does not support bias ATM
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', REGULAR_SEQLEN_2K)
    @pytest.mark.parametrize('seqlen_k', REGULAR_SEQLEN_2K)
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 2 else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_op_bwd_with_matrix_bias(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
        causal = False
        bias_type = 'matrix'
        '''
        _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
        '''
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        _test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 0:  # Make the loading faster
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
    @pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [4, 143, 2048])
    @pytest.mark.parametrize('seqlen_k', [4, 127, 579, 2048])
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_gqa(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        _test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 1:  # Make the loading faster
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', PRIME_SEQLEN_Q)
    @pytest.mark.parametrize('seqlen_k', PRIME_SEQLEN_K)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_irregulars(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        _test_op_bwd(request, args, device=torch_gpu)

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

@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_logsumexp_scaling(dtype):
    REF_VALUE = 2.79018449783325195
    device = 'cuda'
    q = torch.eye(16, device=device, dtype=dtype).reshape((1,1,16,16))
    k = torch.eye(16, device=device, dtype=dtype).reshape((1,1,16,16))
    v = torch.eye(16, device=device, dtype=dtype).reshape((1,1,16,16))
    b = None
    causal = False
    sm_scale = 1.0 / math.sqrt(16)
    dropout_p = 0.0

    ext = AttentionExtraArgs(return_encoded_softmax=False,
                             autotune=False,
                             return_autotune=False,
                             return_logsumexp=True)
    tri_out, _, L = attention(q, k, v, b, causal, sm_scale, dropout_p, ext)
    ref_tensor = torch.full_like(L, REF_VALUE)
    assert torch.allclose(L, ref_tensor)

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
    is_allclose, adiff, grads_allclose, grads_adiff = ctx.validate_with_reference(tri_out, ctx.dout_tensors, use_adiff_entry=use_adiff_entry)
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
    D_HEAD = 64
    N_HEADS = 8
    seqlen_q = 256
    seqlen_k = 256
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
