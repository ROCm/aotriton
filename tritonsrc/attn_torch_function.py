#!/usr/bin/env python

import torch
import triton
import triton.language as tl
from flash import attn_fwd as bare_attn_fwd
from flash import (
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq
)

VERBOSE=False

def is_power_of_two(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0

def is_supported_by_tl_dot(n: int) -> bool:
    return is_power_of_two(n) and n >= 16

TRITON_CONFIG_LIST_FWD = [
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': True}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'pre_load_v': False}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4, 'pre_load_v': False}, num_stages=1, num_warps=4),
   ]

'''
# For faster debugging of backward autotune
TRITON_CONFIG_LIST_FWD = [
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2, 'pre_load_v': True}, num_stages=1, num_warps=4),
   ]
'''

@triton.autotune(
   configs=TRITON_CONFIG_LIST_FWD,
   key=['seqlen_q', 'seqlen_k', 'STAGE'],
)
@triton.jit
def tuned_attn_fwd(
    Q, K, V, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    seqlen_q,
    seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    encoded_softmax,
    STAGE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
):
    bare_attn_fwd(
            Q, K, V, sm_scale, M, Out,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_oz, stride_oh, stride_om, stride_on,
            seqlen_q,
            seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset_base,
            encoded_softmax,
            STAGE,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            )

TRITON_CONFIG_LIST_BWD_LARGE_BLOCK = [
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 0}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 2}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 4}, num_stages=1, num_warps=4),
]

TRITON_CONFIG_LIST_BWD_SMALL_BLOCK = [
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 0}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 2}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 3}, num_stages=1, num_warps=4),
       triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 4}, num_stages=1, num_warps=4),
]

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_LARGE_BLOCK,
   key=['seqlen_q', 'seqlen_k'],
)
@triton.jit
def large_tuned_bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    bare_bwd_kernel_dk_dv(Q, K, V, sm_scale, Out, DO,
            DK, DV,
            L,
            D,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            seqlen_q, seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset_base,
            BLOCK_M, BLOCK_DMODEL,
            BLOCK_N,
            CAUSAL,
            ENABLE_DROPOUT)
@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_SMALL_BLOCK,
   key=['seqlen_q', 'seqlen_k'],
)
@triton.jit
def small_tuned_bwd_kernel_dk_dv(
    Q, K, V, sm_scale, Out, DO,
    DK, DV,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    bare_bwd_kernel_dk_dv(Q, K, V, sm_scale, Out, DO,
            DK, DV,
            L,
            D,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            seqlen_q, seqlen_k,
            dropout_p,
            philox_seed,
            philox_offset_base,
            BLOCK_M, BLOCK_DMODEL,
            BLOCK_N,
            CAUSAL,
            ENABLE_DROPOUT)

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_LARGE_BLOCK,
   key=['seqlen_q', 'seqlen_k'],
)
@triton.jit
def large_tuned_bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    bare_bwd_kernel_dq(Q, K, V, sm_scale, Out, DO,
        DQ,
        L,
        D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        seqlen_q, seqlen_k,
        dropout_p,
        philox_seed,
        philox_offset_base,
        BLOCK_M, BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT)

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_SMALL_BLOCK,
   key=['seqlen_q', 'seqlen_k'],
)
@triton.jit
def small_tuned_bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO,
    DQ,
    L,
    D,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    seqlen_q, seqlen_k,
    dropout_p,
    philox_seed,
    philox_offset_base,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
):
    bare_bwd_kernel_dq(Q, K, V, sm_scale, Out, DO,
        DQ,
        L,
        D,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vk, stride_vn,
        seqlen_q, seqlen_k,
        dropout_p,
        philox_seed,
        philox_offset_base,
        BLOCK_M, BLOCK_DMODEL,
        BLOCK_N,
        CAUSAL,
        ENABLE_DROPOUT)

class _attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p, return_encoded_softmax,
                autotune=False, return_autotune=False):
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        o = torch.empty_like(q)
        if torch.version.hip is None:
            BLOCK_M = 128
            BLOCK_N = 64 if Lk <= 64 else 32
            num_stages = 4 if Lk <= 64 else 3
            num_warps = 4 if Lk <= 64 else 8

        stage = 3 if causal else 1
        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            q.shape[1],
            q.shape[0],
        )
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if return_encoded_softmax:
            encoded_softmax = torch.zeros((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=_attention.DEBUG_MASK_DTYPE)
        else:
            encoded_softmax = None
        if False or VERBOSE:
            print(f'{q.shape=}')
            print(f'{k.shape=}')
            print(f'{v.shape=}')
            print(f'{o.shape=}')
            print(f'{q.data_ptr()=:x}')
            print(f'{k.data_ptr()=:x}')
            print(f'{v.data_ptr()=:x}')
            print(f'{M.data_ptr()=:x}')
            print(f'{o.data_ptr()=:x}')
            print(f'{stage=}')
            print(f'seqlen_q={q.shape[2]}')
            print(f'seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        philox_seed = 114514
        philox_offset = 1919810
        MAX_BLOCK_M = 128 if dropout_p == 0 else 64
        MAX_BLOCK_N = 32 if dropout_p == 0 else 32
        MAX_BLOCK_M = MAX_BLOCK_M if is_supported_by_tl_dot(seqlen_q) else 1
        MAX_BLOCK_N = MAX_BLOCK_N if is_supported_by_tl_dot(seqlen_k) else 1

        if autotune:
            tuned_attn_fwd[grid](
                q, k, v, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                seqlen_q=q.shape[2],
                seqlen_k=k.shape[2],
                dropout_p=dropout_p,
                philox_seed=philox_seed,
                philox_offset_base=philox_offset,
                encoded_softmax=encoded_softmax,
                BLOCK_DMODEL=Lk,
                STAGE=stage,
                ENABLE_DROPOUT=dropout_p > 0.0,
                RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
            )
        else:
            bare_attn_fwd[grid](
                q, k, v, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                seqlen_q=q.shape[2],
                seqlen_k=k.shape[2],
                dropout_p=dropout_p,
                philox_seed=philox_seed,
                philox_offset_base=philox_offset,
                encoded_softmax=encoded_softmax,
                STAGE=stage,
                BLOCK_M=min(MAX_BLOCK_M, q.shape[2], k.shape[2]),
                BLOCK_DMODEL=Lk,
                BLOCK_N=min(MAX_BLOCK_N, q.shape[2], k.shape[2]),
                pre_load_v=False,
                ENABLE_DROPOUT=dropout_p > 0.0,
                RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
            )

        ctx.autotune = autotune
        ctx.return_autotune = return_autotune
        if autotune and return_autotune:
            ## restore the grid for bwd kernel
            best_config = tuned_attn_fwd.get_best_config()
            # print(f'{best_config=}')
            # print(f'{dir(best_config)=}')
            # print(f'{str(best_config)=}')
            print("Best config")
            for key, value in best_config.kwargs.items():
                print('\t', key, '=', value)
            print(f'{str(best_config)=}')
            # block_m = int(best_config.__str__().split(",")[0].split("BLOCK_M:")[1])
            block_m = int(best_config.kwargs['BLOCK_M'])
            print(f'{block_m=}')
            BATCH = q.shape[0]
            N_HEADS = q.shape[1]
            D_HEAD = q.shape[3]
            inputs = {
                'Q.shape' : list(q.shape),
                'Q.dtype' : str(q.dtype),
                'N_HEADS' : N_HEADS,
                'D_HEAD' : D_HEAD,
                'seqlen_q' : seqlen_q,
                'seqlen_k' : seqlen_k,
                'STAGE' : stage,
                'RETURN_ENCODED_SOFTMAX': encoded_softmax is not None,
                'BLOCK_DMODEL' : Lk,
                'ENABLE_DROPOUT' : dropout_p > 0.0,
            }
            tuned_kernel = dict(best_config.kwargs)
            compiler_options = {
                'num_warps' : best_config.num_warps,
                'num_stages': best_config.num_stages,
            }
            tuning_result = {
                'kernel_name' : 'attn_fwd',
                'inputs' : inputs,
                'tuned_kernel' : tuned_kernel,
                'compiler_options' : compiler_options,
            }
        else:
            tuning_result = None
            block_m = min(128, q.shape[2], k.shape[2])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[1], q.shape[0])
        ctx.save_for_backward(q, k, v, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.tuning_result = [tuning_result] if tuning_result is not None else None
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward(ctx, do, _, fwd_tuning_result):
        q, k, v, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        do = do.contiguous()
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        seqlen_q = q.shape[2]
        seqlen_k = k.shape[2]
        MAX_BLOCK = 64 if ctx.dropout_p == 0 else 16
        # BLOCK = min(seqlen_q, seqlen_k, q.shape[-1], MAX_BLOCK)
        # BLOCK = BLOCK if is_supported_by_tl_dot(seqlen_q) and is_supported_by_tl_dot(seqlen_k) else 1
        if not ctx.autotune:
            BLOCK = 16 # FIXME: Variable block size
        else:
            BLOCK = 128
        return_autotune = ctx.tuning_result is not None

        grid_preprocess = lambda META: (
            do.shape[0] * do.shape[1] * triton.cdiv(do.shape[2], META['BLOCK_M']),
            1,
            1
        )
        bare_bwd_preprocess[grid_preprocess](
            o, do, delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        if False or VERBOSE:
            print(f'{q.shape=} {q.stride()=}')
            print(f'{k.shape=} {k.stride()=}')
            print(f'{v.shape=} {v.stride()=}')
            print(f'{o.shape=} {o.stride()=}')
            print(f'{dq.shape=} {dq.stride()=}')
            print(f'{dk.shape=} {dk.stride()=}')
            print(f'{dv.shape=} {dv.stride()=}')
            print(f'{do.shape=} {do.stride()=}')
            print(f'{L=} {L.shape=}')
            print(f'{delta=}')
            print(f'{BLOCK=}')
        dq = torch.zeros_like(q)
        if ctx.autotune:
            use_small_block = ctx.dropout_p > 0.0
            if use_small_block:
                # DQ_BLOCK_M = min(seqlen_q, BLOCK)
                BLOCK_M = 32
                BLOCK_N = 16
                tuned_bwd_kernel_dk_dv = small_tuned_bwd_kernel_dk_dv
                tuned_bwd_kernel_dq = small_tuned_bwd_kernel_dq
            else:
                BLOCK_M = 128
                BLOCK_N = 64
                tuned_bwd_kernel_dk_dv = large_tuned_bwd_kernel_dk_dv
                tuned_bwd_kernel_dq = large_tuned_bwd_kernel_dq
        else:
            BLOCK_M = 32
            BLOCK_N = 16
            assert BLOCK_M < seqlen_q, f'{BLOCK_M=} < {seqlen_q=}'
            assert BLOCK_N < seqlen_k, f'{BLOCK_N=} < {seqlen_k=}'
        # debug_mask = torch.zeros((q.shape[0], q.shape[1], seqlen_q, seqlen_k), device=q.device, dtype=ctx.encoded_softmax.dtype)
        grid_dk_dv = lambda META: (
            triton.cdiv(seqlen_k, META['BLOCK_N']),
            ctx.grid[1], # Total number of batches (incluing nheads)
            1
        )
        assert v.stride() == dv.stride(), f'V and DV should have the same strides, but {v.stride()=} {dv.stride()=}'
        assert q.stride() == do.stride(), f'Q and DO should have the same strides, but {q.stride()=} {do.stride()=}'
        if k.requires_grad and v.requires_grad:
            if ctx.autotune:
                tuned_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    # debug_mask=debug_mask,
                    BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                )
                if return_autotune:
                    dkdv_best_config = tuned_bwd_kernel_dk_dv.get_best_config()
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'seqlen_q': seqlen_q,
                        'seqlen_k': seqlen_k,
                        'BLOCK_DMODEL' : ctx.BLOCK_DMODEL,
                        'CAUSAL'  : ctx.causal,
                        'ENABLE_DROPOUT' : ctx.dropout_p > 0.0,
                    }
                    tuned_kernel = dict(dkdv_best_config.kwargs)
                    compiler_options = {
                        'num_warps' : dkdv_best_config.num_warps,
                        'num_stages': dkdv_best_config.num_stages,
                    }
                    tuning_result = {
                        'kernel_name' : 'bwd_kernel_dk_dv',
                        'inputs' : inputs,
                        'tuned_kernel' : tuned_kernel,
                        'compiler_options' : compiler_options,
                    }
                    ctx.tuning_result.append(tuning_result)
                    print(f'{id(ctx.tuning_result)=}')
            else:
                bare_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    # debug_mask=debug_mask,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                    CAUSAL=ctx.causal,
                    num_warps=4,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                )
        # mask_allclose = torch.allclose(debug_mask < 0, ctx.encoded_softmax < 0)
        if False:
            mask_allclose = torch.allclose(torch.abs(debug_mask), torch.abs(ctx.encoded_softmax)) # Stores QK
            if not mask_allclose:
                torch.set_printoptions(linewidth=200, threshold=2000)
                import sys
                print(f'bwd mask: {torch.abs(debug_mask[:,:,:2,16:])}')
                print(f'fwd mask: {torch.abs(ctx.encoded_softmax[:,:,:2,16:])}')
                print(f'Full bwd mask: {debug_mask[0,0]}')
                print(f'Full fwd mask: {ctx.encoded_softmax[0,0]}')
                print(f'Full mask div: {debug_mask[0,0] / ctx.encoded_softmax[0,0]}')
                print(f'Full dv: {dv}')
                if seqlen_q == 32:
                    print(f'2nd block bwd mask: {debug_mask[0,0, 16:]}')
                    print(f'2nd block fwd mask: {ctx.encoded_softmax[0,0, 16:]}')
            # print(f'Full q: {q}', file=sys.stderr)
            # assert mask_allclose
        grid_dq = lambda META: (
            triton.cdiv(seqlen_q, META['BLOCK_M']),
            ctx.grid[1], # Total number of batches (incluing nheads)
            1
        )
        if q.requires_grad:
            if ctx.autotune:
                tuned_bwd_kernel_dq[grid_dq](
                    q, k, v, ctx.sm_scale,
                    o, do,
                    dq,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                )
                if return_autotune:
                    dq_best_config = tuned_bwd_kernel_dq.get_best_config()
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'seqlen_q': seqlen_q,
                        'seqlen_k': seqlen_k,
                        'BLOCK_DMODEL' : ctx.BLOCK_DMODEL,
                        'CAUSAL'  : ctx.causal,
                        'ENABLE_DROPOUT' : ctx.dropout_p > 0.0,
                    }
                    tuned_kernel = dict(dq_best_config.kwargs)
                    compiler_options = {
                        'num_warps' : dq_best_config.num_warps,
                        'num_stages': dq_best_config.num_stages,
                    }
                    tuning_result = {
                        'kernel_name' : 'bwd_kernel_dq',
                        'inputs' : inputs,
                        'tuned_kernel' : tuned_kernel,
                        'compiler_options' : compiler_options,
                    }
                    ctx.tuning_result.append(tuning_result)
            else:
                bare_bwd_kernel_dq[grid_dq](
                    q, k, v, ctx.sm_scale,
                    o, do,
                    dq,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    seqlen_q=seqlen_q,
                    seqlen_k=seqlen_k,
                    dropout_p=ctx.dropout_p,
                    philox_seed=ctx.philox_seed,
                    philox_offset_base=ctx.philox_offset,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=ctx.BLOCK_DMODEL,
                    CAUSAL=ctx.causal,
                    num_warps=4, waves_per_eu=1,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                )
        # print(h.asm["ttgir"])
        return dq, dk, dv, None, None, None, None, None, None

attention = _attention.apply
