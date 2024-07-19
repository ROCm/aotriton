import triton
import triton.language as tl
from masked_load_store import mstore2d
from bwd_inner_dkdv import bwd_kernel_dk_dv
from bwd_inner_dq import bwd_kernel_dq

@triton.jit
def attn_bwd(
    Q, K, V, B, sm_scale, Out, DO,
    DK, DV, DQ, DB,
    L, D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvk, stride_dvn,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
    num_head_q : 'i32',
    num_head_k : 'i32',
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens : 'i32',   # set num_seqlens to zero to ignore cu_seqlens_q/k
    max_seqlen_q, # and use max_seqlen_q/k for all seqlen_q/k
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)
    qk_scale = sm_scale * 1.44269504089

    off_h = tl.program_id(1) # head index
    off_z = tl.program_id(2) # batch index, for varlen it indicates index in cu_seqlens_q/k
    # bhid = off_z
    # off_chz = (bhid * N_CTX).to(tl.int64)
    # adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    batch_index = off_z

    if num_seqlens > 0:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        batch_index = 0

    if num_seqlens < 0:  # for padded seqlen
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
        # Varlen, but padded to Rank 4 tensor
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        batch_index = off_z

    off_zh = batch_index * num_head_q + off_h * 1

    # offset pointers for batch/head
    q_offset = off_h * stride_qh + batch_index * stride_qz + cu_seqlens_q_start * stride_qm
    Q += q_offset
    k_offset = off_h * stride_kh + batch_index * stride_kz + cu_seqlens_k_start * stride_kn
    K += k_offset
    v_offset = off_h * stride_vh + batch_index * stride_vz + cu_seqlens_k_start * stride_vk
    V += v_offset
    do_offset = off_h * stride_oh + batch_index * stride_oz + cu_seqlens_q_start * stride_om
    DO += do_offset
    dk_offset = off_h * stride_dkh + batch_index * stride_dkz + cu_seqlens_k_start * stride_dkn
    DK += dk_offset
    dv_offset = off_h * stride_dvh + batch_index * stride_dvz + cu_seqlens_k_start * stride_dvk
    DV += dv_offset
    dq_offset = off_h * stride_dqh + batch_index * stride_dqz + cu_seqlens_q_start * stride_dqm
    DQ += dq_offset
    # M += off_chz
    # D += off_chz
    L += off_zh * max_seqlen_q
    D += off_zh * max_seqlen_q

    # offs_k = tl.arange(0, BLOCK_DMODEL)

    # TODO alibi
    # if USE_ALIBI:
    #     a_offset = bhid
    #     alibi_slope = tl.load(alibi_slopes + a_offset)
    # else:
    #     alibi_slope = None
    alibi_slope = None

    #
    # dk dv section
    #
    start_n = pid * BLOCK_N1
    # This assignment is important. It is what allows us to pick the diagonal
    # blocks. Later, when we want to do the lower triangular, we update start_m
    # after the first dkdv call.
    start_m = start_n if CAUSAL else 0

    if start_n < seqlen_k:
        MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR
        # offs_n = start_n + tl.arange(0, BLOCK_N1)

        dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
        dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

        K_block_ptr = tl.make_block_ptr(
            base=K,
            shape=(seqlen_k, head_dim),
            strides=(stride_kn, stride_kk),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N1, BLOCK_DMODEL),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V,
            shape=(seqlen_k, head_dim),
            strides=(stride_vk, stride_vn),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N1, BLOCK_DMODEL),
            order=(1, 0),
        )

        # load K and V: they stay in SRAM throughout the inner loop for dkdv.
        k = tl.load(K_block_ptr)
        k = (k * qk_scale).to(K_block_ptr.type.element_ty)
        v = tl.load(V_block_ptr)

        if CAUSAL:
            num_steps = BLOCK_N1 // MASK_BLOCK_M1
            # compute dK and dV for blocks close to the diagonal that need to be masked
            dk, dv = bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope,
                                      DO, L, D,
                                      stride_qm, stride_qk,
                                      stride_om, stride_ok,
                                      seqlen_q,
                                      seqlen_k,
                                      head_dim,
                                      MASK_BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,
                                      start_n, start_m, num_steps,
                                      MASK=True, PADDED_HEAD=PADDED_HEAD)
            start_m += num_steps * MASK_BLOCK_M1

        # compute dK and dV for blocks that don't need masking further from the diagonal
        num_steps = (seqlen_q - start_m) // BLOCK_M1  # loop over q

        # tl.device_print('num_steps', num_steps)
        # tl.device_print('MASK_BLOCK_M1', MASK_BLOCK_M1)
        # tl.device_print('BLOCK_N1', BLOCK_N1)
        dk, dv = bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope,
                                  DO, L, D,
                                  stride_qm, stride_qk,
                                  stride_om, stride_ok,
                                  seqlen_q,
                                  seqlen_k,
                                  head_dim,
                                  BLOCK_M1, BLOCK_N1, BLOCK_DMODEL,
                                  start_n, start_m, num_steps,
                                  MASK=False, PADDED_HEAD=PADDED_HEAD)

        # DV_block_ptrs = tl.make_block_ptr(base=DV, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
        #                                   offsets=(start_n, 0), block_shape=(BLOCK_N1, BLOCK_DMODEL), order=(1, 0))
        # tl.store(DV_block_ptrs, dv.to(v.dtype))
        # if pid == 0:
        #     tl.device_print('dv', dv)
        mstore2d(dv.to(v.dtype),
                 BLOCK_N1,
                 BLOCK_DMODEL,
                 o_base=DV,
                 o_start_row=start_n,
                 o_start_col=0,
                 o_rows=seqlen_k,
                 o_cols=head_dim,
                 stride_row=stride_dvk,
                 stride_col=stride_dvn)

        # DK_block_ptrs = tl.make_block_ptr(base=DK, shape=(seqlen_k, head_dim), strides=(stride_vk, stride_vn),
        #                                   offsets=(start_n, 0), block_shape=(BLOCK_N1, BLOCK_DMODEL), order=(1, 0))
        # tl.device_print('dk', dk)
        # tl.store(DK_block_ptrs, (dk * sm_scale).to(k.dtype))
        mstore2d((dk * sm_scale).to(k.dtype),
                 BLOCK_N1,
                 BLOCK_DMODEL,
                 o_base=DK,
                 o_start_row=start_n,
                 o_start_col=0,
                 o_rows=seqlen_k,
                 o_cols=head_dim,
                 stride_row=stride_dkn,
                 stride_col=stride_dkk)

    #
    # dq section
    #
    start_m = pid * BLOCK_M2
    end_n = start_m + BLOCK_M2 if CAUSAL else seqlen_k  # look over k/v

    if start_m < seqlen_q:
        MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
        offs_m = start_m + tl.arange(0, BLOCK_M2)

        # Q_block_ptr = tl.make_block_ptr(base=Q, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
        #                                 offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
        Q_block_ptr = tl.make_block_ptr(base=Q, shape=(seqlen_q, head_dim), strides=(stride_qm, stride_qk),
                                        offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))

        DO_block_ptr = tl.make_block_ptr(base=DO, shape=(seqlen_q, head_dim), strides=(stride_om, stride_ok),
                                         offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
        q = tl.load(Q_block_ptr)
        q = (q * qk_scale).to(Q_block_ptr.type.element_ty)
        do = tl.load(DO_block_ptr)
        dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)

        m = tl.load(L + offs_m)
        m = m[:, None]

        num_steps = BLOCK_M2 // MASK_BLOCK_N2
        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _attn_bwd_dq, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        if CAUSAL:
            dq = bwd_kernel_dq(dq, q, K, V, alibi_slope,
                               do, m, D,
                               stride_kn, stride_kk,
                               stride_vk, stride_vn,
                               seqlen_q,
                               seqlen_k,
                               head_dim,
                               BLOCK_M2,
                               MASK_BLOCK_N2,
                               BLOCK_DMODEL,
                               start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps,
                               MASK=True, PADDED_HEAD=PADDED_HEAD)
            end_n -= num_steps * MASK_BLOCK_N2
        # stage 2
        num_steps = end_n // BLOCK_N2
        dq = bwd_kernel_dq(dq, q, K, V, alibi_slope,
                           do, m, D,
                           stride_kn, stride_kk,
                           stride_vk, stride_vn,
                           seqlen_q,
                           seqlen_k,
                           head_dim,
                           BLOCK_M2,
                           BLOCK_N2,
                           BLOCK_DMODEL,
                           start_m, end_n - num_steps * BLOCK_N2, num_steps,
                           MASK=False, PADDED_HEAD=PADDED_HEAD)
        # Write back dQ.
        # DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
        #                                  offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
        # dq *= LN2
        # tl.store(DQ_block_ptr, dq.to(q.dtype))
        mstore2d((dq * sm_scale).to(q.dtype),
                 BLOCK_M2,
                 BLOCK_DMODEL,
                 o_base=DQ,
                 o_start_row=start_m,
                 o_start_col=0,
                 o_rows=seqlen_q,
                 o_cols=head_dim,
                 stride_row=stride_dqm,
                 stride_col=stride_dqk)
