#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import triton
import triton.language as tl
from dropout import dropout_mask, dropout_rng, dropout_offsets
from masked_load_store import load_fn

# Helper function, but not always usable due to compiler bugs (esp. used with tl.trans)
@triton.jit
def dot(BLOCK_M : tl.constexpr, QDIM : tl.constexpr, KDIM : tl.constexpr, q, k):
    if BLOCK_M == 1:
        return tl.sum(tl.view(q, [QDIM]) * tl.view(k, [KDIM]))
    else:
        return tl.dot(q, k)

@triton.jit
def bwd_inner_dq(
    # I/O Tensor
    dq,
    DB_block_ptr, store_db,
    # Problem Description
    q, kt_ptrs, k_stride, vt_ptrs, v_stride, B_block_ptr,
    do,
    Di, l_i,
    seqlen_q, seqlen_k, head_dim,
    # Sub-problem range, (lo, hi) specify the range for seqlen_q
    start_q, lo, hi,
    ## Dropout
    ### max_seqlen_k is put in Dropout section because it is not needed by
    ### anything other than dropout
    dropout_p, philox_seed, batch_philox_offset, max_seqlen_k,
    # constexpr starts here
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    # DEBUG_RIGHT: tl.constexpr,
    FULL_BLOCKS: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    # initialize offsets
    offs_q = start_q + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    ld_offs_d = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)

    '''
    if DEBUG_RIGHT and start_q == 0:
        tl.device_print('Right Di', Di)
        tl.device_print('Right l_i', l_i)
    if not DEBUG_RIGHT and start_q == 0:
        tl.device_print('Wrong Di', Di)
        tl.device_print('Wrong l_i', l_i)
    '''

    kt_ptrs += lo * k_stride
    vt_ptrs += lo * v_stride
    if BIAS_TYPE == 1:
        B_block_ptr = tl.advance(B_block_ptr, (0, lo))
        DB_block_ptr = tl.advance(DB_block_ptr, (0, lo))

    '''
           K1   K2      (d)V      dO
    Q1    qk11 qk12     (d)v1     dO1
    Q2    qk21 qk22     (d)v2     dO2

    QK: (seqlen_q, seqlen_k)
    dO: (seqlen_q, hdim)
    dV: (seqlen_k, hdim)
    '''
    # tl.device_print('bwd_inner_dq', FULL_BLOCKS, lo, hi)
    for start_k in range(lo, hi, BLOCK_N):
        offs_k_curr = offs_k[None, :] + start_k # (1, BLOCK_N)
        # -- load k, v --
        # shape = (BLOCK_DMODEL, BLOCK_N), offs = (0, BLOCK_N * iter) = (0, start_k)
        # kt = tl.load(K_block_ptr)
        # vt = tl.load(V_block_ptr)
        if not FULL_BLOCKS:
            kt = load_fn(kt_ptrs, ld_offs_d, offs_k + start_k, head_dim, seqlen_k)
        else:
            kt = load_fn(kt_ptrs, ld_offs_d, None, head_dim, seqlen_k)

        # TODO: pre_load_vt
        if not FULL_BLOCKS:
            vt = load_fn(vt_ptrs, ld_offs_d, offs_k + start_k, head_dim, seqlen_k)
        else:
            vt = load_fn(vt_ptrs, ld_offs_d, None, head_dim, seqlen_k)
        # -- compute qk ----
        # q.offs = (start_m, 0), k.offs = (0, start_k)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, q, kt)
        if not FULL_BLOCKS:
            k_boundary = tl.full((BLOCK_M, ), seqlen_k, dtype=tl.int32)
            mask = offs_k_curr < k_boundary[:, None]
            qk = tl.where(mask, qk, float("-inf"))
        if CAUSAL:
            # qk = tl.where(offs_q[:, None] >= (offs_k[None, :] + start_k), qk, float("-inf"))
            qk = tl.where(offs_q[:, None] >= offs_k_curr, qk, float("-inf"))
        if BIAS_TYPE == 0:
            pass
        elif BIAS_TYPE == 1:
            '''
            if q_padded and k_padded:  # CAVEAT: using "or" disables the partial boundary_check branches
                bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            elif q_padded:
                bias = tl.load(B_block_ptr, boundary_check=(0,), padding_option="zero")
            elif k_padded:
                bias = tl.load(B_block_ptr, boundary_check=(1,), padding_option="zero")
            else:
                bias = tl.load(B_block_ptr)
            '''
            # FIXME: Must use boundary_check uncondtionally.
            # The optimized tl.load above causes nan for some reason
            bias = tl.load(B_block_ptr, boundary_check=(0,1), padding_option="zero")
            qk += bias * 1.44269504089
        else:
            tl.static_assert(False, f'Unsupported BIAS_TYPE {BIAS_TYPE}')
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += dot(BLOCK_M, BLOCK_DMODEL, BLOCK_DMODEL, do, vt)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_q * max_seqlen_k + start_k
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, max_seqlen_k)
            dp = tl.where(keep, dp / (1 - dropout_p), 0)
        # compute ds = p * (dp - delta[:, None])
        ds = p * (dp - Di[:, None])
        # compute dq. Unfortunately we cannot avoid transpose here as this loop
        # uses k both normal and transpose.

        if BLOCK_M == 1:
            dq += tl.view(kt, [BLOCK_DMODEL]) * ds.to(q.type.element_ty)
        else:
            # ds.shape = (BLOCK_M, BLOCK_N), kt.shape = (BLOCK_DMODEL, BLOCK_N)
            dq = tl.dot(ds.to(q.type.element_ty), tl.trans(kt), acc=dq) # (BLOCK_M, BLOCK_DMODEL)

        '''
        if DEBUG_RIGHT:
            if not FULL_BLOCKS and start_k == 0:
                tl.device_print('Right dq first', dq)
                tl.device_print('Right ds first', ds)
            if not FULL_BLOCKS and start_k == 16:
                tl.device_print('Right dq second', dq)
                tl.device_print('Right ds second', ds)
                # tl.device_print('Right p second', p)
                # tl.device_print('Right dp second', dp)
                tl.device_print('Right do second', do)
                tl.device_print('Right vt second', vt)
        if not DEBUG_RIGHT:
            if not FULL_BLOCKS and start_k == 0:
                tl.device_print('Wrong dq first', dq)
                tl.device_print('Wrong ds first', ds)
            if not FULL_BLOCKS and start_k == 16:
                tl.device_print('Wrong dq second', dq)
                tl.device_print('Wrong ds second', ds)
                # tl.device_print('Wrong p second', p)
                # tl.device_print('Wrong dp second', dp)
                tl.device_print('Wrong do second', do)
                tl.device_print('Wrong vt second', vt)
        #     # tl.device_print('ds semi', ds)
        #     tl.device_print('kt semi', kt)
        '''

        if BIAS_TYPE == 1:
            if store_db:
                tl.store(DB_block_ptr, ds.to(DB_block_ptr.type.element_ty), boundary_check=(0,1))
        # update pointers
        # Keep the block ptr as comment
        # K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        # V_block_ptr = tl.advance(V_block_ptr, (0, BLOCK_N))
        kt_ptrs += BLOCK_N * k_stride
        vt_ptrs += BLOCK_N * v_stride
        if BIAS_TYPE == 1:
            B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_N))
            DB_block_ptr = tl.advance(DB_block_ptr, (0, BLOCK_N))
    return dq
