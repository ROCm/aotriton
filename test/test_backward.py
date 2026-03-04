#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pyaotriton import get_name_suffix
assert get_name_suffix() != "tRiToN_tEsTeR", ("AOTriton is compiled with suffix 'tRiToN_tEsTeR'. "
                                              "This is a signature for AOTriton built to test Triton compiler, "
                                              "which has fewer functionals selected "
                                              "and should not be used for general tests.")
import pytest
from _core_test_backward import (
    ALL_INT_HEADDIMS,
    ALL_TUP_HEADDIMS,
    REGULAR_SEQLEN,
    REGULAR_SEQLEN_2K,
    PRIME_SEQLEN_Q,
    PRIME_SEQLEN_K,
    PRIME_SEQLEN_Q_1K,
    PRIME_SEQLEN_K_1K,
    FOR_RELEASE,
    BWD_IMPL,
    DTYPES,
    BWDOP_ids,
    fmt_nheads,
    fmt_hdim,
    gpufilelock,
    torch_gpu,
    test_logsumexp_scaling,
    core_test_op_bwd,
    core_test_large_bf16_nan_values,
)

if FOR_RELEASE >= 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5, (10, 2)] if BWD_IMPL != 2 else [5], ids=fmt_nheads)
    @pytest.mark.parametrize('D_HEAD', [8, 64, 184, (24, 152), (120, 8)], ids=fmt_hdim)
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
        core_test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
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
        core_test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 0 and BWD_IMPL != 2:  # AITER ASM does not support bias ATM
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
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
        core_test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 0 and BWD_IMPL != 2:  # AITER ASM does not expose GQA
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [(16, 8), (10, 2)])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
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
        core_test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 1:  # Make the loading faster
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
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
        if bias_type is not None and BWD_IMPL == 2:
            pytest.skip("Bias is not supported in AITER ASM backend")
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=torch_gpu)

if FOR_RELEASE > 2:  # Testing hdim_qk != hdim_vo
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_TUP_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', PRIME_SEQLEN_Q_1K)
    @pytest.mark.parametrize('seqlen_k', PRIME_SEQLEN_K_1K)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False])
    @pytest.mark.parametrize('bias_type', [None, 'matrix'], ids=['BiasOff', 'BiasOn'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_hdim_qk_ne_vo(request, torch_gpu, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=torch_gpu)

@pytest.mark.parametrize('D_HEAD', [16])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_large_bf16_nan_values(BWDOP, D_HEAD):
    core_test_large_bf16_nan_values(D_HEAD)

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
    core_test_op_bwd(args)

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
    core_test_op_bwd(args)

if __name__ == '__main__':
    main2()
    # main_npz()
