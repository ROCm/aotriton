import triton
import triton.language as tl

@triton.jit
def bwd_kernel_dq(dq, q, K, V, alibi_slope,
                  do, m, D,
                  stride_kn, stride_kk,
                  stride_vk, stride_vn,
                  seqlen_q,
                  seqlen_k,
                  head_dim,
                  BLOCK_M2: tl.constexpr,
                  BLOCK_N2: tl.constexpr,
                  BLOCK_DMODEL: tl.constexpr,
                  # Filled in by the wrapper.
                  start_m, start_n, num_steps,
                  MASK: tl.constexpr,
                  PADDED_HEAD: tl.constexpr,
                  ):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    # offs_k = tl.arange(0, BLOCK_DMODEL)
    KT_block_ptr = tl.make_block_ptr(base=K, shape=(head_dim, seqlen_k), strides=(stride_kk, stride_kn),
                                     offsets=(0, start_n), block_shape=(BLOCK_DMODEL, BLOCK_N2), order=(0, 1))
    VT_block_ptr = tl.make_block_ptr(base=V, shape=(head_dim, seqlen_k), strides=(stride_vn, stride_vk),
                                     offsets=(0, start_n), block_shape=(BLOCK_DMODEL, BLOCK_N2), order=(0, 1))
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        if PADDED_HEAD:
            kT = tl.load(KT_block_ptr, boundary_check=(0,1), padding_option="zero")
        else:
            kT = tl.load(KT_block_ptr)
        qk = tl.dot(q, kT)
        # if alibi_slope is not None:
        #     alibi_block = compute_alibi_block(alibi_slope, N_CTX, N_CTX, offs_m, offs_n)
        #     qk += alibi_block * 1.44269504089

        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        if PADDED_HEAD:
            vT = tl.load(VT_block_ptr, boundary_check=(0,1), padding_option="zero")
        else:
            vT = tl.load(VT_block_ptr)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(KT_block_ptr.type.element_ty)
        # Compute dQ.0.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        KT_block_ptr = tl.advance(KT_block_ptr, (0, step_n))
        VT_block_ptr = tl.advance(VT_block_ptr, (0, step_n))
    return dq

