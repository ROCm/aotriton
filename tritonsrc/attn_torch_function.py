#!/usr/bin/env python
# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from collections import namedtuple
import copy
import torch
import triton
import triton.language as tl
from flash import (
    debug_fill_dropout_rng as bare_debug_fill_dropout_rng,
    attn_fwd as bare_attn_fwd,
    bwd_preprocess as bare_bwd_preprocess,
    bwd_kernel_dk_dv as bare_bwd_kernel_dk_dv,
    bwd_kernel_dq as bare_bwd_kernel_dq,
)
from tuned_bwd import (
    tuned_bwd_kernel_dk_dv,
    tuned_bwd_kernel_dq,
)
from sized_tuned_bwd import (
    sized_tuned_bwd_kernel_dk_dv,
    sized_tuned_bwd_kernel_dq,
)

# Note: we don't use Enum class because accessing the integer requires using
#       `.value` property, which makes the code verbose.
class CausalType:
    NONE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2

class BiasType:
    NONE = 0
    MATRIX = 1
    VECTOR = 2  # CAVEAT: Unsupported in kernel

class PersistentType:
    NONE = 0
    FIXED = 1
    DYNAMIC = 2

def factor_head_dim(head_dim, n_pieces=3):
    ret = [0] * 3
    Lk = head_dim
    for i in range(n_pieces):
        max_po2 = 2 ** (Lk.bit_length() - 1)
        # Technically Triton now supports all power-of-two, lowering to 1
        # But PyTorch pads all inputs to multiple of 8.
        # In addition we do not have the capability to support that many choices
        max_po2 = max(8, max_po2)
        ret[i] = max_po2
        # print(f"\t{i=}: {Lk=} {max_po2=} left: {Lk - max_po2}")
        Lk -= max_po2
        if Lk <= 0:
            break
    while sum(ret) < head_dim:
        ret[-1] *= 2
        ret = sorted(ret, reverse=True)
    return ret

AttentionExtraArgs = namedtuple('AttentionExtraArgs',
        ['return_encoded_softmax',
         'autotune',
         'return_autotune',
         'fillnan',
         'report_best_config',
         ],
        defaults=[False, False, False, False, None])

VERBOSE=False
DEFAULT_PHILOX_SEED = 0x1BF52
DEFAULT_PHILOX_OFFSET_1 = 0x1D4000
DEFAULT_PHILOX_OFFSET_2 = 0x000B42
DEFAULT_PHILOX_OFFSET = DEFAULT_PHILOX_OFFSET_1 + DEFAULT_PHILOX_OFFSET_2

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
   key=['max_seqlen_q', 'max_seqlen_k', 'CAUSAL'],
)
@triton.jit
def tuned_attn_fwd(
    Q, K, V, B, sm_scale, M, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_bz, stride_bh, stride_bm, stride_bn,
    stride_oz, stride_oh, stride_om, stride_on,
    num_head_q,
    num_head_k,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,
    max_seqlen_q,
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed_ptr,
    philox_offset1,
    philox_offset2,
    philox_seed_output,
    philox_offset_output,
    encoded_softmax,
    CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    pre_load_v: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    RETURN_ENCODED_SOFTMAX: tl.constexpr,
    PADDED_HEAD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
):
    bare_attn_fwd(
            Q, K, V, B, sm_scale, M, Out,
            stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vk, stride_vn,
            stride_bz, stride_bh, stride_bm, stride_bn,
            stride_oz, stride_oh, stride_om, stride_on,
            num_head_q,
            num_head_k,
            cu_seqlens_q,
            cu_seqlens_k,
            num_seqlens,
            max_seqlen_q,
            max_seqlen_k,
            head_dim,
            dropout_p,
            philox_seed_ptr,
            philox_offset1,
            philox_offset2,
            philox_seed_output,
            philox_offset_output,
            encoded_softmax,
            CAUSAL,
            BLOCK_M,
            BLOCK_DMODEL,
            BLOCK_N,
            pre_load_v,
            ENABLE_DROPOUT,
            RETURN_ENCODED_SOFTMAX,
            PADDED_HEAD,
            BIAS_TYPE=BIAS_TYPE,
            )


TRITON_CONFIG_LIST_BWD_FUSED = []
for BLOCK_M1 in [16, 32, 64]:
    for BLOCK_N1 in [16, 32, 64, 128, 256]:
        if BLOCK_N1 % BLOCK_M1 != 0:
            continue
        for BLOCK_M2 in [16, 32]:
            for BLOCK_N2 in [16, 32]:
                if BLOCK_M2 % BLOCK_N2 != 0:
                    continue
                dic = {'BLOCK_M1': BLOCK_M1, 'BLOCK_N1': BLOCK_N1}
                dic['BLOCK_M2'] = BLOCK_M2
                dic['BLOCK_N2'] = BLOCK_N2
                dic['BLK_SLICE_FACTOR'] = 2
                for waves_per_eu in range(0, 4+1):
                    dic['waves_per_eu'] = waves_per_eu
                    for num_stages in [0, 1]:
                        for num_warps in [1,2,4,8]:
                            cfg = triton.Config(dic, num_stages=num_stages, num_warps=num_warps)
                            TRITON_CONFIG_LIST_BWD_FUSED.append(cfg)

@triton.autotune(
   configs=TRITON_CONFIG_LIST_BWD_FUSED,
   key=['max_seqlen_q', 'max_seqlen_k', 'head_dim'],
)
@triton.jit
def tuned_attn_bwd(
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
    num_head_q,
    num_head_k,
    cu_seqlens_q,
    cu_seqlens_k,
    num_seqlens,
    max_seqlen_q, # and use max_seqlen_q/k for all seqlen_q/k
    max_seqlen_k,
    head_dim,
    dropout_p,
    philox_seed_ptr,
    philox_offset1,
    philox_offset2,
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
    bare_attn_bwd(
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
            num_head_q,
            num_head_k,
            cu_seqlens_q,
            cu_seqlens_k,
            num_seqlens,
            max_seqlen_q, # and use max_seqlen_q/k for all seqlen_q/k
            max_seqlen_k,
            head_dim,
            dropout_p,
            philox_seed_ptr,
            philox_offset_base,
            BLOCK_DMODEL,
            CAUSAL,
            ENABLE_DROPOUT,
            PADDED_HEAD,
            BIAS_TYPE,
            BLOCK_M1,
            BLOCK_N1,
            BLOCK_M2,
            BLOCK_N2,
            BLK_SLICE_FACTOR,
            )

class _attention(torch.autograd.Function):

    # DEBUG_MASK_DTYPE = torch.int32
    DEBUG_MASK_DTYPE = torch.float32

    @staticmethod
    def forward(ctx, q, k, v, b, causal, sm_scale, dropout_p,
                attn_extra_args=AttentionExtraArgs()):
        return_encoded_softmax, autotune, return_autotune = attn_extra_args[:3]
        dtype = q.dtype
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        head_dim_factors = factor_head_dim(Lk)
        head_dim_rounded = sum(head_dim_factors)
        padded_head = head_dim_rounded != Lk
        # assert not padded_head, f"sum({head_dim_factors=}) = {sum(head_dim_factors)} != {Lk=}"
        num_head_q = q.shape[1]
        num_head_k = k.shape[1]
        max_seqlen_q = q.shape[2]
        max_seqlen_k = k.shape[2]
        o = torch.empty_like(q)

        grid = lambda META: (
            triton.cdiv(q.shape[2], META['BLOCK_M']),
            num_head_q,
            q.shape[0],
        )
        null_tensor = torch.empty((0), device=q.device, dtype=torch.int32)
        M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if attn_extra_args.fillnan:
            for t in (o, M):
                t.fill_(float('nan'))
        if return_encoded_softmax:
            encoded_softmax = torch.ones((q.shape[0], q.shape[1], q.shape[2], k.shape[2]), device=q.device, dtype=q.dtype)
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
            print(f'max_seqlen_q={q.shape[2]}')
            print(f'max_seqlen_k={k.shape[2]}')
            print(f'{v.data_ptr()=:x}')
            print(f'{v.stride(1)=:x}')
            print(f'{v.data_ptr() + q.shape[0] * q.shape[1] * v.stride(1)=:x}')
            if encoded_softmax is not None:
                print(f'{encoded_softmax.shape=} {encoded_softmax.dtype=}')

        if dropout_p > 0.0:
            philox_seed = torch.tensor([DEFAULT_PHILOX_SEED], device=q.device, dtype=torch.uint64)
            philox_offset1 = torch.tensor([DEFAULT_PHILOX_OFFSET_1], device=q.device, dtype=torch.uint64)
            philox_offset2 = DEFAULT_PHILOX_OFFSET_2
            philox_seed_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
            philox_offset_output = torch.tensor([0], device=q.device, dtype=torch.uint64)
        else:
            u64nulltensor = torch.empty([0], device=q.device, dtype=torch.uint64)
            philox_seed = u64nulltensor
            philox_offset1 = u64nulltensor
            philox_offset2 = 0
            philox_seed_output = u64nulltensor
            philox_offset_output = u64nulltensor

        if b is None:
            b = torch.empty((0,0,0,0), device=q.device, dtype=q.dtype)
            BIAS_TYPE = BiasType.NONE
        else:
            BIAS_TYPE = BiasType.MATRIX

        # TODO alibi_slopes
        alibi_slopes = torch.empty((0,0), device=q.device, dtype=q.dtype)

        # TODO: int8
        q_descale = k_descale = p_scale = p_descale = v_descale = 0

        use_small_block = dropout_p > 0.0 or BIAS_TYPE != 0 or return_encoded_softmax
        use_medium_block = False # reserved
        if use_small_block:
            BLOCK_M = 64
            BLOCK_N = 32
        elif use_medium_block:
            BLOCK_M = 64
            BLOCK_N = 64
        else:
            BLOCK_M = 128
            BLOCK_N = 64
        if dtype == torch.float32:
            BLOCK_M //= 2

        if autotune:
            tuned_attn_fwd[grid](
                q, k, v, b, alibi_slopes, sm_scale, M, o,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                num_head_q=num_head_q,
                num_head_k=num_head_k,
                cu_seqlens_q=null_tensor,
                cu_seqlens_k=null_tensor,
                num_seqlens=0,
                max_seqlen_q=q.shape[2],
                max_seqlen_k=k.shape[2],
                head_dim=Lk,
                dropout_p=dropout_p,
                philox_seed_ptr=philox_seed,
                philox_offset1=philox_offset1,
                philox_offset2=philox_offset2,
                philox_seed_output=philox_seed_output,
                philox_offset_output=philox_offset_output,
                encoded_softmax=encoded_softmax,
                CAUSAL=causal,
                BLOCK_DMODEL=head_dim_rounded,
                ENABLE_DROPOUT=dropout_p > 0.0,
                RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
                PADDED_HEAD=padded_head,
                BIAS_TYPE=BIAS_TYPE,
            )
        else:
            RETURN_ENCODED_SOFTMAX=encoded_softmax is not None
            print(f'{BLOCK_M=} {BLOCK_N=} {RETURN_ENCODED_SOFTMAX=} seqlen_q={q.shape[2]} seqlen_k={k.shape[2]}',
                    flush=True)
            print(f'{q.data_ptr()=:x} {k.data_ptr()=:x} {v.data_ptr()=:x} {b.data_ptr()=:x} {M.data_ptr()=:x} {o.data_ptr()=:x}', flush=True)
            if RETURN_ENCODED_SOFTMAX:
                print(f'{encoded_softmax.data_ptr()=:x}', flush=True)
            print(f'{q.shape=} {k.shape=} {v.shape=} {b.shape=} {M.shape=} {o.shape=}', flush=True)
            print(f'{q.stride()=} {k.stride()=} {v.stride()=} {b.stride()=} {M.stride()=} {o.stride()=}', flush=True)
            bare_attn_fwd[grid](
                # Basic SDPA
                q, k, v, b, alibi_slopes, sm_scale, M, o,
                q_descale, k_descale, p_scale, p_descale, v_descale,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *o.stride(),
                *b.stride(),
                *alibi_slopes.stride(),
                # MQA/GQA
                Num_head_q=num_head_q,
                Num_head_k=num_head_k,
                # Varlen
                Num_seqlens=0,
                cu_seqlens_q=null_tensor,
                cu_seqlens_k=null_tensor,
                Max_seqlen_q=q.shape[2],
                Max_seqlen_k=k.shape[2],
                # Head Dimensions
                BLOCK_DMODEL=head_dim_rounded,
                Head_dim=Lk,
                PADDED_HEAD=padded_head,
                # droput and PRNG
                ENABLE_DROPOUT=dropout_p > 0.0,
                dropout_p=dropout_p,
                philox_seed_ptr=philox_seed,
                philox_offset1=philox_offset1,
                philox_offset2=philox_offset2,
                philox_seed_output=philox_seed_output,
                philox_offset_output=philox_offset_output,
                RETURN_ENCODED_SOFTMAX=encoded_softmax is not None,
                encoded_softmax=encoded_softmax,
                # Causal
                CAUSAL_TYPE=CausalType.TOP_LEFT if causal else CausalType.NONE,
                # bias
                BIAS_TYPE=BIAS_TYPE,
                # INT8
                INT8=False,
                INT8_KV=False,
                USE_P_SCALE=False,
                # Alibi
                USE_ALIBI=False,
                # Persistent related arguments
                PERSISTENT_TYPE=PersistentType.NONE,
                persistent_atomic_counter=0,
                Num_CU=40,
                GRID_CU_MULTIP=2,
                Batch=q.shape[0],
                # Performance
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                PRE_LOAD_V=False,
                num_stages=1,
            )

        ctx.autotune = autotune
        ctx.return_autotune = return_autotune
        if autotune and return_autotune:
            ## restore the grid for bwd kernel
            best_config = tuned_attn_fwd.get_best_config()
            tuning_result = copy.deepcopy(best_config)
            block_m = int(best_config.kwargs['BLOCK_M'])
            """
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
                'max_seqlen_q' : max_seqlen_q,
                'max_seqlen_k' : max_seqlen_k,
                'CAUSAL' : causal,
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
            """
        else:
            tuning_result = None
            block_m = min(128, q.shape[2], k.shape[2])
        grid = (triton.cdiv(q.shape[2], block_m), q.shape[1], q.shape[0])
        # print(f'{M=}')
        # print(f'{M.shape=}')
        ctx.save_for_backward(q, k, v, b, o, M)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.head_dim = Lk
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.philox_seed = philox_seed_output
        ctx.philox_offset = philox_offset_output
        ctx.encoded_softmax = encoded_softmax # FIXME: for debugging only
        ctx.bias_type = BIAS_TYPE
        ctx.tuning_result = [('attn_fwd', tuning_result)] if tuning_result is not None else None
        ctx.attn_extra_args = attn_extra_args
        if ctx.tuning_result is not None:
            for kernel_name, best in ctx.tuning_result:
                print(f'{kernel_name=} {best.kwargs=} {best.num_warps=} {best.num_stages=}')
        return o, encoded_softmax, ctx.tuning_result

    @staticmethod
    def backward_split(ctx, do, _, fwd_tuning_result):
        q, k, v, b, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv and Lk == ctx.head_dim
        head_dim_factors = factor_head_dim(Lk)
        head_dim_rounded = sum(head_dim_factors)
        padded_head = head_dim_rounded != ctx.head_dim
        attn_extra_args = ctx.attn_extra_args
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b)
        delta = torch.empty_like(L)
        if attn_extra_args.fillnan:
            for t in (dq, dk, dv, db, delta):
                t.fill_(float('nan'))
        null_tensor = torch.empty((0), device=q.device, dtype=torch.int32)
        num_head_q = int(q.shape[1])
        num_head_k = int(k.shape[1])
        max_seqlen_q = q.shape[2]
        max_seqlen_k = k.shape[2]
        MAX_BLOCK = 64 if ctx.dropout_p == 0 else 16
        # BLOCK = min(max_seqlen_q, max_seqlen_k, q.shape[-1], MAX_BLOCK)
        # BLOCK = BLOCK if is_supported_by_tl_dot(max_seqlen_q) and is_supported_by_tl_dot(max_seqlen_k) else 1
        if not ctx.autotune:
            BLOCK = 16 # FIXME: Variable block size
        else:
            BLOCK = 128
        return_autotune = ctx.tuning_result is not None

        grid_prep = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
        bare_bwd_preprocess[grid_prep](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            max_seqlen_q,
            Lk,
            BLOCK_M=BLOCK, D_HEAD=head_dim_rounded,
            PADDED_HEAD=padded_head, # FIXME: irregular head dimension
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

        use_small_block = ctx.dropout_p > 0.0
        use_medium_block = ctx.bias_type != 0
        # Profiling shows (16, 16) is optimal solution for most bwd configurations
        BLOCK_M = 16
        BLOCK_N = 16
        # if use_small_block:
        #     # DQ_BLOCK_M = min(max_seqlen_q, BLOCK)
        #     BLOCK_M = 32
        #     BLOCK_N = 16
        # elif use_medium_block:
        #     BLOCK_M = 64
        #     BLOCK_N = 32
        # else:
        #     BLOCK_M = 64
        #     BLOCK_N = 64
        # if q.dtype == torch.float32:
        #     BLOCK_M = max(16, BLOCK_M // 2)
        #     BLOCK_N = max(16, BLOCK_N // 2)
        # debug_mask = torch.zeros((q.shape[0], q.shape[1], max_seqlen_q, max_seqlen_k), device=q.device, dtype=ctx.encoded_softmax.dtype)
        grid_dk_dv = lambda META: (
            triton.cdiv(max_seqlen_k, META['BLOCK_N']),
            num_head_k,
            q.shape[0],
        )
        stride_dbz, stride_dbh, stride_dbm, stride_dbn = db.stride()
        if db.numel() == 0 or not b.requires_grad:
            # Passing all zeros to indicate no elements
            stride_dbz, stride_dbh, stride_dbm, stride_dbn = 0,0,0,0
        else:
            db.fill_(float('nan'))
        print(f'backward {ctx.bias_type=} {ctx.autotune=} {BLOCK_M=} {BLOCK_N=} {stride_dbz=} {stride_dbh=} {stride_dbm=} {stride_dbn=}')
        if k.requires_grad and v.requires_grad:
            if ctx.autotune:
                tuned_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed_ptr=philox_seed,
                    philox_offset1=philox_offset,
                    philox_offset2=0,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                report = attn_extra_args.report_best_config
                if report is not None:
                    best = copy.deepcopy(tuned_bwd_kernel_dk_dv.best_config)
                    attn_extra_args.report_best_config('bwd_kernel_dk_dv', best)
                if return_autotune:
                    dkdv_best_config = copy.deepcopy(sized_tuned_bwd_kernel_dk_dv.get_best_config())
                    # BLOCK_M/N are missing with sized_tuned_bwd_kernel_*
                    dkdv_best_config.kwargs['BLOCK_M'] = BLOCK_M
                    dkdv_best_config.kwargs['BLOCK_N'] = BLOCK_N
                    tuning_result = copy.deepcopy(dkdv_best_config)
                    """
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'max_seqlen_q': max_seqlen_q,
                        'max_seqlen_k': max_seqlen_k,
                        'head_dim' : ctx.BLOCK_DMODEL,
                        'BLOCK_DMODEL' : head_dim_rounded,
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
                    """
                    ctx.tuning_result.append(('bwd_kernel_dk_dv', tuning_result))
                    print(f'{id(ctx.tuning_result)=}')
            else:
                print('Running bare_bwd_kernel_dk_dv')
                bare_bwd_kernel_dk_dv[grid_dk_dv](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed_ptr=philox_seed,
                    philox_offset1=philox_offset,
                    philox_offset2=0,
                    # debug_mask=debug_mask,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    num_warps=2, waves_per_eu=1,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                print('bare_bwd_kernel_dk_dv Done')
        # print(f"{dq.stride()=}", flush=True)
        # print(f"{dq.data_ptr()=:x}", flush=True)
        # print(f"{dk.stride()=}", flush=True)
        # print(f"{dk.data_ptr()=:x}", flush=True)
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
                if max_seqlen_q == 32:
                    print(f'2nd block bwd mask: {debug_mask[0,0, 16:]}')
                    print(f'2nd block fwd mask: {ctx.encoded_softmax[0,0, 16:]}')
            # print(f'Full q: {q}', file=sys.stderr)
            # assert mask_allclose
        grid_dq = lambda META: (
            triton.cdiv(max_seqlen_q, META['BLOCK_M']),
            num_head_q,
            q.shape[0],
        )
        if q.requires_grad:
            if ctx.autotune:
                tuned_bwd_kernel_dq[grid_dq](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed_ptr=philox_seed,
                    philox_offset1=philox_offset,
                    philox_offset2=0,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                report = attn_extra_args.report_best_config
                if report is not None:
                    best = copy.deepcopy(tuned_bwd_kernel_dq.best_config)
                    attn_extra_args.report_best_config('bwd_kernel_dq', best)
                if return_autotune:
                    dq_best_config = copy.deepcopy(sized_tuned_bwd_kernel_dq.get_best_config())
                    # BLOCK_M/N are missing with sized_tuned_bwd_kernel_*
                    dq_best_config.kwargs['BLOCK_M'] = BLOCK_M
                    dq_best_config.kwargs['BLOCK_N'] = BLOCK_N
                    tuning_result = dq_best_config
                    """
                    inputs = {
                        'Q.shape' : list(q.shape),
                        'Q.dtype' : str(q.dtype),
                        'N_HEADS' : q.shape[1],
                        'max_seqlen_q': max_seqlen_q,
                        'max_seqlen_k': max_seqlen_k,
                        'head_dim' : ctx.BLOCK_DMODEL,
                        'BLOCK_DMODEL' : head_dim_rounded,
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
                    """
                    ctx.tuning_result.append(('bwd_kernel_dq', tuning_result))
            else:
                print('Running bare_bwd_kernel_dq')
                bare_bwd_kernel_dq[grid_dq](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed_ptr=philox_seed,
                    philox_offset1=philox_offset,
                    philox_offset2=0,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    num_warps=4, waves_per_eu=1,
                    num_stages=1,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                )
                print('bare_bwd_kernel_dq Done')
        # print(h.asm["ttgir"])
        return dq, dk, dv, None if db.numel() == 0 else db, None, None, None, None, None, None, None

    @staticmethod
    def backward_fused(ctx, do, _, fwd_tuning_result):
        q, k, v, b, o, L = ctx.saved_tensors
        # if q.shape[-1] <= 32:
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv and Lk == ctx.head_dim
        head_dim_rounded = 2 ** (ctx.head_dim - 1).bit_length()
        head_dim_rounded = max(16, head_dim_rounded)
        padded_head = head_dim_rounded != ctx.head_dim
        attn_extra_args = ctx.attn_extra_args
        philox_seed = ctx.philox_seed
        philox_offset = ctx.philox_offset

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b)
        delta = torch.empty_like(L)
        if attn_extra_args.fillnan:
            for t in (dq, dk, dv, db, delta):
                t.fill_(float('nan'))
        null_tensor = torch.empty((0), device=q.device, dtype=torch.int32)
        batch = q.shape[0]
        num_head_q = q.shape[1]
        num_head_k = q.shape[2]
        max_seqlen_q = q.shape[2]
        max_seqlen_k = k.shape[2]
        MAX_BLOCK = 64 if ctx.dropout_p == 0 else 16
        stride_dbz, stride_dbh, stride_dbm, stride_dbn = db.stride()
        if db.numel() == 0 or not b.requires_grad:
            # Passing all zeros to indicate no elements
            stride_dbz, stride_dbh, stride_dbm, stride_dbn = 0,0,0,0
        else:
            db.fill_(float('nan'))
        BLOCK = 128
        grid_prep = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])
        bare_bwd_preprocess[grid_prep](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            max_seqlen_q,
            Lk,
            BLOCK_M=BLOCK, D_HEAD=head_dim_rounded,
            PADDED_HEAD=padded_head, # FIXME: irregular head dimension
        )
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
        BLK_SLICE_FACTOR = 2
        # BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 16, 16, 16, 16
        # BLK_SLICE_FACTOR = 1
        grid = lambda META: (max(triton.cdiv(max_seqlen_k, META['BLOCK_M1']), triton.cdiv(max_seqlen_q, META['BLOCK_M2'])), num_head_q, batch)
        if ctx.autotune:
            tuned_attn_bwd[grid](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv, dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,  # db may be empty
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=q.shape[2],
                    max_seqlen_k=k.shape[2],
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed_ptr=philox_seed,
                    philox_offset1=philox_offset,
                    philox_offset2=0,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                    )
            report = attn_extra_args.report_best_config
            if report is not None:
                best = copy.deepcopy(tuned_attn_bwd.best_config)
                attn_extra_args.report_best_config('attn_bwd', best)
        else:
            bare_attn_bwd[grid](
                    q, k, v, b, ctx.sm_scale,
                    o, do,
                    dk, dv, dq, db,
                    L, delta,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    b.stride(0), b.stride(1), b.stride(2), b.stride(3),
                    do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                    dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                    dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                    dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                    stride_dbz, stride_dbh, stride_dbm, stride_dbn,  # db may be empty
                    num_head_q=num_head_q,
                    num_head_k=num_head_k,
                    cu_seqlens_q=null_tensor,
                    cu_seqlens_k=null_tensor,
                    num_seqlens=0,
                    max_seqlen_q=q.shape[2],
                    max_seqlen_k=k.shape[2],
                    head_dim=Lk,
                    dropout_p=ctx.dropout_p,
                    philox_seed_ptr=philox_seed,
                    philox_offset1=philox_offset,
                    philox_offset2=0,
                    BLOCK_DMODEL=head_dim_rounded,
                    CAUSAL=ctx.causal,
                    ENABLE_DROPOUT=ctx.dropout_p > 0.0,
                    PADDED_HEAD=padded_head,
                    BIAS_TYPE=ctx.bias_type,
                    BLOCK_M1=BLOCK_M1,
                    BLOCK_N1=BLOCK_N1,
                    BLOCK_M2=BLOCK_M2,
                    BLOCK_N2=BLOCK_N2,
                    BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
                    )
        return dq, dk, dv, None if db.numel() == 0 else db, None, None, None, None, None, None, None

    backward = backward_split

attention = _attention.apply

def debug_fill_dropout_rng(dropout_rng, philox_seed, philox_offset):
    BLOCK_M = 64
    BLOCK_N = 32
    BATCH, N_HEADS, seqlen_q, seqlen_k = dropout_rng.size()
    grid_rng = lambda META: (
        triton.cdiv(seqlen_q, META['BLOCK_M']),
        N_HEADS,
        BATCH,
    )
    r = dropout_rng
    bare_debug_fill_dropout_rng[grid_rng](r,
            r.stride(0), r.stride(1), r.stride(2), r.stride(3),
            seqlen_q, seqlen_k,
            philox_seed,
            philox_offset,
            BLOCK_M, BLOCK_N,
            num_stages=1)
