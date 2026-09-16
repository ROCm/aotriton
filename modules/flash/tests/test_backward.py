#!/usr/bin/env python
# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# FIXME: MUST Import torch before pyaotriton now. pyaotriton.so may resolve to
# system RCCL which is incompatible with torch's RCCL
import torch
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
    PRIME_HEADDIMS,
    core_test_logsumexp_scaling,
    core_test_matrix_bias_fwd_bwd_symmetry,
    core_test_op_bwd,
    core_test_large_bf16_nan_values,
    core_test_bottom_right_fully_masked_rows,
)
from _common_test import ALL_LAYOUTS, StorageLayout

if FOR_RELEASE >= 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5, (10, 2)] if BWD_IMPL != 'aiter' else [5], ids=fmt_nheads)
    @pytest.mark.parametrize('D_HEAD', [8, 64, 184, (24, 152), (120, 8), (64, 32)], ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [11, 523, 2048])
    @pytest.mark.parametrize('seqlen_k', [31, 337, 1063])
    # pairing causal and bias_type to eliminate programmatic skips.
    # Gated on BWD_IMPL != 'aiter' for the same reason line 71 below is: AITER ASM
    # does not support bias, so the matrix-bias pair has to come out of the
    # list rather than be skipped inside the test -- the point of pairing is
    # that there is no programmatic skip left here.
    @pytest.mark.parametrize('causal,bias_type',
                             [(False, None), (False, 'matrix'), (True, None)]
                             if BWD_IMPL != 'aiter' else [(False, None), (True, None)],
                             ids=['CausalOff-BiasOff', 'CausalOff-BiasOn', 'CausalOn-BiasOff']
                             if BWD_IMPL != 'aiter' else ['CausalOff-BiasOff', 'CausalOn-BiasOff'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 'aiter' else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_fast(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 0:
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', REGULAR_SEQLEN)
    @pytest.mark.parametrize('seqlen_k', REGULAR_SEQLEN)
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 'aiter' else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1', 'l2'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_regular_bwd(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 0 and BWD_IMPL != 'aiter':  # AITER ASM does not support bias ATM
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', ALL_INT_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', REGULAR_SEQLEN_2K)
    @pytest.mark.parametrize('seqlen_k', REGULAR_SEQLEN_2K)
    @pytest.mark.parametrize('dropout_p', [0.0, 0.5] if BWD_IMPL != 'aiter' else [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_op_bwd_with_matrix_bias(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, sm_scale, dropout_p, dtype, storage_flip):
        causal = False
        bias_type = 'matrix'
        '''
        _scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True
        '''
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE > 0 and BWD_IMPL != 'aiter':  # AITER ASM does not expose GQA
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
    def test_gqa(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip):
        bias_type = None
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE >= 0:
    # The 8xD input contract, exercised. See PRIME_HEADDIMS in
    # _core_test_backward.py for why these were disabled and what changed.
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5])
    @pytest.mark.parametrize('D_HEAD', PRIME_HEADDIMS, ids=fmt_hdim)
    @pytest.mark.parametrize('seqlen_q', [257])
    @pytest.mark.parametrize('seqlen_k', [571])
    @pytest.mark.parametrize('causal', [False, True], ids=['CausalOff', 'CausalOn'])
    @pytest.mark.parametrize('dropout_p', [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    @pytest.mark.parametrize('storage_flip', [False, True])
    @pytest.mark.parametrize('bias_type', [None], ids=['BiasOff'])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_prime_hdim(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

if FOR_RELEASE >= 0:
    # **Every tensor in a different memory layout, with the head dim off the
    # 8-multiple grid.** test_prime_hdim above proves the 8xD contract holds;
    # this proves the descriptor that implements it is derived from the right
    # tensor's stride, which is a question only a permuted layout can ask.
    #
    # `_slab_span_elems` spans a `(batch, head)` slab as
    # `(rows-1)*stride_seq + ceil8(hdim)`, built per tensor from THAT tensor's
    # own stride. Under BHSD every tensor agrees, so reading the wrong one is
    # invisible -- and BHSD plus the single transposition `storage_flip` reaches
    # were the only layouts anything exercised.
    #
    # Sensitivity: end a slab at `hdim` instead and the hardware drops the dword
    # holding columns `hdim-1` and `hdim`, taking the real column with it on the
    # last row of every slab. Measured at these shapes: `O[.., seqlen_q-1,
    # hdim-1]` never stored, `dQ[.., seqlen_q-1, hdim-1]` and `dK/dV[..,
    # seqlen_kv-1, hdim-1]` garbage, and the NaN left in O reaches every element
    # of dK through delta.
    LAYOUT_CASES = [StorageLayout.round_robin(case) for case in range(len(ALL_LAYOUTS))]

    # BATCH and N_HEADS must BOTH be > 1. At 1 the corresponding stride is
    # arbitrary and unconstrained, so the six permutations collapse into fewer
    # than six distinct stride patterns and the test stops asking its question.
    @pytest.mark.parametrize('BATCH', [3])
    @pytest.mark.parametrize('N_HEADS', [5], ids=fmt_nheads)
    # One prime on each side of the point where the D axis stops being a single
    # tile: 53 rides one block, 179 is loaded as a composed 128+64 pair, so both
    # the simple and the composed store path meet a row whose last dword straddles
    # the slab bound. Both are odd, which is what puts the last real column in the
    # same dword as the first pad column.
    @pytest.mark.parametrize('D_HEAD', [53, 179], ids=fmt_hdim)
    # Off the block grid in both directions, as test_prime_hdim uses them: the
    # defect is on the LAST row of a slab, so a seqlen that divides the block size
    # evenly would never produce a ragged one.
    @pytest.mark.parametrize('seqlen_q', [257])
    @pytest.mark.parametrize('seqlen_k', [571])
    # Paired rather than crossed, the way test_fast pairs them, so there is no
    # programmatic skip: `causal and bias_type is not None` is rejected by
    # _scaled_dot_product_attention, and AITER ASM has no bias at all. Bias is in
    # here because it has a descriptor of its own (`_bias_slab_num_records_bytes`)
    # and an innermost axis that is the KV sequence rather than a head dim.
    @pytest.mark.parametrize('causal,bias_type',
                             [(False, None), (False, 'matrix'), (True, None)]
                             if BWD_IMPL != 'aiter' else [(False, None), (True, None)],
                             ids=['CausalOff-BiasOff', 'CausalOff-BiasOn', 'CausalOn-BiasOff']
                             if BWD_IMPL != 'aiter' else ['CausalOff-BiasOff', 'CausalOn-BiasOff'])
    @pytest.mark.parametrize('dropout_p', [0.0])
    @pytest.mark.parametrize('dtype', DTYPES)
    @pytest.mark.parametrize('sm_scale', ['l1'])
    # The `storage_flip` slot takes either spelling; _do_test_op_bwd dispatches on
    # the type. See _common_test.StorageLayout for the round robin these are.
    @pytest.mark.parametrize('storage_flip', LAYOUT_CASES,
                             ids=[f'Layouts{case}' for case in range(len(ALL_LAYOUTS))])
    @pytest.mark.parametrize('BWDOP', BWDOP_ids)
    def test_memory_layouts(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

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
    def test_irregulars(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if bias_type is not None and BWD_IMPL == 'aiter':
            pytest.skip("Bias is not supported in AITER ASM backend")
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

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
    def test_hdim_qk_ne_vo(request, gpu_id, BWDOP, BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type):
        if bias_type is not None and (seqlen_q > 2048 or seqlen_k > 2048):
            pytest.skip("Skip large UT with bias to avoid OOM")
        args = (BATCH, N_HEADS, D_HEAD, seqlen_q, seqlen_k, causal, sm_scale, dropout_p, dtype, storage_flip, bias_type)
        core_test_op_bwd(request, args, device=gpu_id)

@pytest.mark.parametrize('D_HEAD', [16])
@pytest.mark.parametrize('BWDOP', BWDOP_ids)
def test_large_bf16_nan_values(BWDOP, D_HEAD):
    core_test_large_bf16_nan_values(D_HEAD)

# Defined here rather than imported from _core_test_backward, so that conftest.py's
# _FILE_ORDER sees this file. The gpu_id is the second half of the fix:
# core_test_logsumexp_scaling hardcodes device='cuda', so without the context manager
# it runs on the default device regardless of which GPU this xdist worker leased.
@pytest.mark.parametrize('dtype', DTYPES)
def test_logsumexp_scaling(gpu_id, dtype):
    with torch.cuda.device(gpu_id):
        core_test_logsumexp_scaling(dtype)

# Defined here for the same two reasons as test_logsumexp_scaling above.
@pytest.mark.parametrize('bias_val', [0.0, 4.0, 16.0, -16.0, 64.0])
@pytest.mark.parametrize('dtype', DTYPES)
def test_matrix_bias_fwd_bwd_symmetry(gpu_id, dtype, bias_val):
    with torch.cuda.device(gpu_id):
        core_test_matrix_bias_fwd_bwd_symmetry(dtype, bias_val)

# ROCm/aotriton#235. Ungated (level 0) on purpose: it guards a silent
# wrong-answer path -- bottom-right causal with seqlen_q > seqlen_k returned
# unwritten Out/LSE rows once the tile count exceeded the workgroup count --
# and it costs one small forward per repeat. This is the shape PyTorch's
# memory-efficient attention takes for causal_lower_right, where the
# corresponding upstream test is currently skipped on ROCm.
#
# Not parametrized over BWDOP: the defect is in attn_fwd, and the backward
# backend is irrelevant to it.
def test_bottom_right_fully_masked_rows(gpu_id):
    with torch.cuda.device(gpu_id):
        core_test_bottom_right_fully_masked_rows(f'cuda:{gpu_id}')

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
