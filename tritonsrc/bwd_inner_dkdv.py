import triton
import triton.language as tl
from masked_load_store import mload1d, mload2d

@triton.jit
def bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope,
                     DO, M, D,
                     # shared by Q/K/V/DO.
                     stride_qm, stride_qk,
                     stride_om, stride_ok,
                     seqlen_q,
                     seqlen_k,
                     head_dim,
                     BLOCK_M1: tl.constexpr,
                     BLOCK_N1: tl.constexpr,
                     BLOCK_DMODEL: tl.constexpr,
                     # Filled in by the wrapper.
                     start_n, start_m, num_steps,
                     MASK: tl.constexpr,
                     PADDED_HEAD: tl.constexpr,
                     ):
    # offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # offs_k = tl.arange(0, BLOCK_DMODEL)
    QT_block_ptr = tl.make_block_ptr(base=Q, shape=(head_dim, seqlen_q), strides=(stride_qk, stride_qm),
                                     offsets=(0, start_m), block_shape=(BLOCK_DMODEL, BLOCK_M1), order=(0, 1))
    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(seqlen_q, head_dim), strides=(stride_om, stride_ok),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M1, BLOCK_DMODEL), order=(1, 0))
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        if PADDED_HEAD:
            qT = tl.load(QT_block_ptr, boundary_check=(0,1), padding_option="zero")
        else:
            qT = tl.load(QT_block_ptr)

        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        # m = tl.load(M + offs_m)
        m = mload1d(BLOCK_M1, i_base=M, i_start=curr_m, i_nums=seqlen_q)
        kqT = tl.dot(k, qT)
        # if alibi_slope is not None:
        #     alibi_block = compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, True)
        #     kqT += alibi_block * 1.44269504089

        pT = tl.math.exp2(kqT - m[None, :])
        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(DO_block_ptr)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(DO_block_ptr.dtype.element_ty)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do))  # dp += tl.dot(do, vt)
        # dsT = pT * (dpT - Di[None, :])
        dsT = (dpT - Di[None, :]) * pT  # ds = p * (dp - Di) # (BLOCK_M, BLOCK_N)
        # dk += tl.dot(dsT, tl.trans(qT))
        dk += tl.dot(dsT.to(QT_block_ptr.dtype.element_ty), tl.trans(qT))  # dk += tl.dot(tl.trans(ds.to(Q_block_ptr.dtype.element_ty)), q) # (BLOCK_N, BLOCK_DMODEL)
        # Increment pointers.
        curr_m += step_m
        QT_block_ptr = tl.advance(QT_block_ptr, (0, step_m))
        DO_block_ptr = tl.advance(DO_block_ptr, (step_m, 0))
    return dk, dv

