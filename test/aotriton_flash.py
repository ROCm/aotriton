# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pyaotriton.v2.flash import (
    attn_fwd as fa_forward,
    attn_bwd as fa_backward,
    attn_fwd_compact_varlen as fa_forward_compact_varlen,
    attn_bwd_compact_varlen as fa_backward_compact_varlen,
    debug_fill_dropout_rng as fa_debug_fill_dropout_rng,
    FwdExtraArguments,
    BwdExtraArguments,
)
from pyaotriton import T1, T2, T4, DType, Stream, hipError_t, get_name_suffix
assert get_name_suffix() != "", ("To run tests, AOTriton must be compiled with suffixes "
                                 "by passing -DAOTRITON_NAME_SUFFIX=SOME_SUFFIX to cmake. "
                                 "Otherwise the AOTriton in-development may have conflicts with "
                                 "AOTriton shipped with PyTorch.")
try:
    from pyaotriton import T0
    PASS_PHILOX_AS_TENSOR = True
except:
    PASS_PHILOX_AS_TENSOR = False
from pyaotriton.v2 import CppTuneSpecialKernelIndex
import os

def cast_dtype(dtype):
    assert not dtype.is_complex
    bits = dtype.itemsize * 8
    if dtype.is_floating_point:
        maintype = 'Float' if 'bfloat' not in str(dtype) else 'BFloat'
    else:
        maintype = 'Int' if 'uint' not in str(dtype) else 'UInt'
    typename = f'k{maintype}{bits}'
    return getattr(DType, typename)

def mk_aotensor(q, if_empty_then_like=None):
    rank = len(q.shape) if q is not None else len(if_empty_then_like.shape)
    if q is not None and len(q.shape) == 1 and q.numel() == 1:
        if PASS_PHILOX_AS_TENSOR:
            return T0(q.data_ptr(), cast_dtype(q.dtype))
        else:
            return q[0]
    elif rank == 1:
        klass = T1
    elif rank == 2:
        klass = T2
    elif rank == 4:
        klass = T4
    else:
        assert False, f'Unsupported tensor rank {rank}, shape {q.shape}'
    if q is None:
        return klass(0, [0] * rank, [0] * rank, cast_dtype(if_empty_then_like.dtype))
    if q is not None:
        assert q.stride(-1) == 1, "AOTriton assumes the last stride of Tensors be 1"
    return klass(q.data_ptr(), tuple(q.size()), q.stride(), cast_dtype(q.dtype))

def attn_fwd(q, k, v, b, sm_scale, M, o,
             dropout_p, philox_seed, philox_offset1, philox_offset2,
             philox_seed_output, philox_offset_output,
             encoded_softmax, is_causal,
             extargs=None):
    extargs = FwdExtraArguments() if extargs is None else extargs
    err = fa_forward(mk_aotensor(q),
                     mk_aotensor(k),
                     mk_aotensor(v),
                     mk_aotensor(b, if_empty_then_like=q),
                     float(sm_scale),
                     mk_aotensor(M),
                     mk_aotensor(o),
                     float(dropout_p),
                     mk_aotensor(philox_seed),
                     mk_aotensor(philox_offset1),
                     philox_offset2,
                     T0(philox_seed_output.data_ptr(), DType.kUInt64),
                     T0(philox_offset_output.data_ptr(), DType.kUInt64),
                     mk_aotensor(encoded_softmax, if_empty_then_like=q),
                     is_causal,
                     Stream(),
                     extargs)
    # print(f'{err=}')
    return err

def attn_bwd(q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, L, delta,
             dropout_p, philox_seed, philox_offset1, philox_offset2, is_causal, extargs=None):
    extargs = BwdExtraArguments() if extargs is None else extargs
    b = mk_aotensor(b, if_empty_then_like=q)
    # print(f'{b=}')
    err = fa_backward(mk_aotensor(q),
                      mk_aotensor(k),
                      mk_aotensor(v),
                      b,
                      float(sm_scale),
                      mk_aotensor(o),
                      mk_aotensor(dout),
                      mk_aotensor(dq),
                      mk_aotensor(dk),
                      mk_aotensor(dv),
                      mk_aotensor(db, if_empty_then_like=q),
                      mk_aotensor(L),
                      mk_aotensor(delta),
                      float(dropout_p),
                      mk_aotensor(philox_seed),
                      mk_aotensor(philox_offset1),
                      philox_offset2,
                      is_causal,
                      Stream(),
                      extargs)
    # print(f'{err=}')
    return err

def debug_fill_dropout_rng(R, philox_seed, philox_offset):
    err = fa_debug_fill_dropout_rng(mk_aotensor(R),
                                    philox_seed,
                                    philox_offset,
                                    Stream())
    # print(f'debug_fill_dropout_rng {err=}')
    return err

def attn_fwd_compact_varlen(q, k, v,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        b, sm_scale, M, o,
        dropout_p, philox_seed, philox_offset1, philox_offset2,
        philox_seed_output, philox_offset_output,
        encoded_softmax, is_causal):
    err = fa_forward_compact_varlen(mk_aotensor(q),
                                    mk_aotensor(k),
                                    mk_aotensor(v),
                                    mk_aotensor(cu_seqlens_q),
                                    mk_aotensor(cu_seqlens_k),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    mk_aotensor(b, if_empty_then_like=q),
                                    float(sm_scale),
                                    mk_aotensor(M),
                                    mk_aotensor(o),
                                    float(dropout_p),
                                    mk_aotensor(philox_seed),
                                    mk_aotensor(philox_offset1),
                                    philox_offset2,
                                    mk_aotensor(philox_seed_output),
                                    mk_aotensor(philox_offset_output),
                                    mk_aotensor(encoded_softmax, if_empty_then_like=q),
                                    is_causal,
                                    Stream())
    # print(f'{err=}')
    return err

def attn_bwd_compact_varlen(q, k, v,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        b, sm_scale, o, dout, dq, dk, dv, db, L, delta,
        dropout_p, philox_seed, philox_offset1, philox_offset2, is_causal):
    b = mk_aotensor(b, if_empty_then_like=q)
    # print(f'{b=}')
    err = fa_backward_compact_varlen(mk_aotensor(q),
                                     mk_aotensor(k),
                                     mk_aotensor(v),
                                     mk_aotensor(cu_seqlens_q),
                                     mk_aotensor(cu_seqlens_k),
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     b,
                                     float(sm_scale),
                                     mk_aotensor(o),
                                     mk_aotensor(dout),
                                     mk_aotensor(dq),
                                     mk_aotensor(dk),
                                     mk_aotensor(dv),
                                     mk_aotensor(db, if_empty_then_like=q),
                                     mk_aotensor(L),
                                     mk_aotensor(delta),
                                     float(dropout_p),
                                     mk_aotensor(philox_seed),
                                     mk_aotensor(philox_offset1),
                                     philox_offset2,
                                     is_causal,
                                     Stream())
    # print(f'{err=}')
    return err
