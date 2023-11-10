#!/usr/bin/env python

"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""
import triton
import triton.language as tl
from fwd_kernel import dropout_mask, dropout_rng

@triton.jit
def bwd_kernel(
    Q, K, V, sm_scale,
    Out, dO,
    dQ, dK, dV,
    L, D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    Z, H,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    # debug_mask,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, # No support for Causal = True in fused kernel for now
    ENABLE_DROPOUT: tl.constexpr,
):
    batch_index = tl.program_id(0)
    qk_scale = sm_scale * 1.44269504
    q_offset = batch_index * stride_qh
    k_offset = batch_index * stride_kh
    k_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    '''
    kT_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    '''
    v_offset = batch_index * stride_vh
    vT_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    dKT_block_ptr = tl.make_block_ptr(
        base=dK + k_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    dVT_block_ptr = tl.make_block_ptr(
        base=dV + v_offset,
        shape=(BLOCK_DMODEL, seqlen_k),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    dV_block_ptr = tl.make_block_ptr(
        base=dV + v_offset,
        shape=(seqlen_k, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    '''
    debug_mask_ptr = tl.make_block_ptr(
        base=debug_mask + batch_index * seqlen_q * seqlen_k,
        shape=(seqlen_q, seqlen_k),
        strides=(seqlen_k, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0)
    )
    '''

    # L.shape = (q.shape[0] * q.shape[1], q.shape[2])
    L_ptr = L + batch_index * seqlen_q
    D_ptr = D + batch_index * seqlen_q
    range_m = tl.arange(0, BLOCK_M)
    batch_philox_offset = philox_offset_base + batch_index * seqlen_q * seqlen_k
    # Note the backward partition tasks vertically
    # For consistency with fwd kernel, this kernel keeps using BLOCK_M to partition Q and BLOCK_N for KV
    for start_n in range(0, seqlen_k, BLOCK_N):
        lo = 0 # Keeps this for future support of Causal
        # dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dkT = tl.zeros([BLOCK_DMODEL, BLOCK_N], dtype=tl.float32)
        # dvT = tl.zeros([BLOCK_DMODEL, BLOCK_N], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        k = tl.load(k_block_ptr) # (BLOCK_N, BLOCK_DMODEL)
        # kT = tl.load(kT_block_ptr) # (BLOCK_DMODEL, BLOCK_N)
        vT = tl.load(vT_block_ptr) # (BLOCK_DMODEL, BLOCK_N)
        Q_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(seqlen_q, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )
        QT_block_ptr = tl.make_block_ptr(
            base=Q + q_offset,
            shape=(BLOCK_DMODEL, seqlen_q),
            strides=(stride_qk, stride_qm),
            offsets=(0, 0),
            block_shape=(BLOCK_DMODEL, BLOCK_M),
            order=(0, 1),
        )
        dO_block_ptr = tl.make_block_ptr(
            base=dO + q_offset,
            shape=(seqlen_q, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )
        # dO^T
        # dOT_block_ptr = tl.make_block_ptr(
        #     base=DO + q_offset,
        #     shape=(BLOCK_DMODEL, seqlen_q),
        #     strides=(stride_qk, stride_qm),
        #     offsets=(0, 0),
        #     block_shape=(BLOCK_DMODEL, BLOCK_M),
        #     order=(0, 1)
        # )
        dQ_block_ptr = tl.make_block_ptr(
            base=dQ + q_offset,
            shape=(seqlen_q, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0)
        )
        # Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
        for start_m in range(lo, seqlen_q, BLOCK_M):
            # ''' qk '''
            # q = tl.load(Q_block_ptr) # (BLOCK_M, BLOCK_DMODEL)
            # qkT = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            # qkT += tl.dot(q, kT) # (BLOCK_M, BLOCK_N)
            ''' kq '''
            qT = tl.load(QT_block_ptr) # (BLOCK_DMODEL, BLOCK_M)
            kqT = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
            kqT += tl.dot(k, qT) # (BLOCK_N, BLOCK_M)
            # tl.store(dQ_block_ptr, kqT.to(dQ.dtype.element_ty)) # DEBUG

            l_i = tl.load(L_ptr + start_m + range_m) # (BLOCK_M), l_i[:, None].shape = (BLOCK_M, 1)
            # ''' dv^T '''
            # p = tl.math.exp2(qkT * qk_scale - l_i[:, None]) # (BLOCK_M, BLOCK_N)
            # doT = tl.load(DOT_block_ptr) # (BLOCK_DMODEL, BLOCK_M)
            # dvT += tl.dot(doT, p) # dV += P^T dO => dV^T += dO^T P, (BLOCK_DMODEL, BLOCK_N)
            ''' dV '''
            pT = tl.math.exp2(kqT * qk_scale - l_i[None, :]) # (BLOCK_N, BLOCK_M)
            do = tl.load(dO_block_ptr) # (BLOCK_M, BLOCK_DMODEL)
            if ENABLE_DROPOUT:
                philox_offset = batch_philox_offset + start_m * seqlen_k + start_n
                '''
                keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int8)
                keep += dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
                tl.store(debug_mask_ptr, dropout_rng(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k))
                keept = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int8)
                keept += tl.trans(keep)
                '''
                keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, seqlen_k)
                # CAVEAT: do NOT update pT, ds needs the original p
                dv += tl.dot(tl.where(tl.trans(keep), pT / (1 - dropout_p), 0).to(do.type.element_ty), do) # (BLOCK_N, BLOCK_DMODEL)
            else:
                dv += tl.dot(pT.to(do.type.element_ty), do) # (BLOCK_N, BLOCK_DMODEL)
            # dv += 1.0 # DEBUG
            # tl.store(dVT_block_ptr, dv.to(dV.type.element_ty)) # DEBUG
            # dp^T
            # Di = tl.load(D_ptr + range_m) # (BLOCK_M), Di[None, :].shape = (1, BLOCK_M)
            # dpT = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32) - Di[None, :]
            # dpT += tl.dot(v, doT) # (BLOCK_N, BLOCK_M)
            ''' dp and p '''
            Di = tl.load(D_ptr + start_m + range_m) # (BLOCK_M), Di[:, None].shape = (BLOCK_M, 1)
            dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            dp += tl.dot(do, vT) # (BLOCK_M, BLOCK_N)
            if ENABLE_DROPOUT:
                dp = tl.where(keep, dp / (1 - dropout_p), 0)
            p = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            p += tl.trans(pT)
            # ds^T
            # CAVEAT, ds requires p BEFORE dropout
            ds = p * (dp - Di[:, None]) * sm_scale
            # ''' dk = dot(ds.T, q) '''
            # dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
            ''' dkT = qT . ds '''
            dkT += tl.dot(qT.to(ds.type.element_ty), ds)
            dq = tl.load(dQ_block_ptr)
            dq += tl.dot(ds.to(Q.dtype.element_ty), k)
            tl.store(dQ_block_ptr, dq)
            # Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
            QT_block_ptr = tl.advance(QT_block_ptr, (0, BLOCK_M))
            dQ_block_ptr = tl.advance(dQ_block_ptr, (BLOCK_M, 0))
            dO_block_ptr = tl.advance(dO_block_ptr, (BLOCK_M, 0))
            # debug_mask_ptr = tl.advance(debug_mask_ptr, (BLOCK_M, 0))
        tl.store(dKT_block_ptr, dkT.to(K.type.element_ty))
        tl.store(dV_block_ptr, dv.to(dV.type.element_ty))
        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_N, 0))
        vT_block_ptr = tl.advance(vT_block_ptr, (0, BLOCK_N))
        dKT_block_ptr = tl.advance(dKT_block_ptr, (0, BLOCK_N))
        dVT_block_ptr = tl.advance(dVT_block_ptr, (0, BLOCK_N))
        dV_block_ptr = tl.advance(dV_block_ptr, (BLOCK_N, 0))
        # debug_mask_ptr = tl.advance(debug_mask_ptr, (-seqlen_q, 0))
        # debug_mask_ptr = tl.advance(debug_mask_ptr, (0, BLOCK_N))

