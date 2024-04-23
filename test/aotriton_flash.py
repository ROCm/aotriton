# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pyaotriton.v2.flash import attn_fwd as fa_forward, attn_bwd as fa_backward
from pyaotriton import T1, T2, T4, DType, Stream

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
    if rank == 1:
        klass = T1
    elif rank == 2:
        klass = T2
    elif rank == 4:
        klass = T4
    else:
        assert False, f'Unsupported tensor rank {rank}, shape {q.shape}'
    if q is None:
        return klass(0, [0] * rank, [0] * rank, cast_dtype(if_empty_then_like.dtype))
    return klass(q.data_ptr(), tuple(q.size()), q.stride(), cast_dtype(q.dtype))

def attn_fwd(q, k, v, b, sm_scale, M, o,
             dropout_p, philox_seed, philox_offset, encoded_softmax, is_causal):
    err = fa_forward(mk_aotensor(q),
                     mk_aotensor(k),
                     mk_aotensor(v),
                     mk_aotensor(b, if_empty_then_like=q),
                     float(sm_scale),
                     mk_aotensor(M),
                     mk_aotensor(o),
                     float(dropout_p),
                     int(philox_seed),
                     int(philox_offset),
                     mk_aotensor(encoded_softmax, if_empty_then_like=q),
                     is_causal,
                     Stream())
    print(f'{err=}')

def attn_bwd(q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, L, delta,
             dropout_p, philox_seed, philox_offset, is_causal):
    b = mk_aotensor(b, if_empty_then_like=q)
    print(f'{b=}')
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
                      int(philox_seed),
                      int(philox_offset),
                      is_causal,
                      Stream())
    print(f'{err=}')
