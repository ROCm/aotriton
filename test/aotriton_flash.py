# Copyright © 2023-2025 Advanced Micro Devices, Inc.
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

AOTRITON_TORCH_ONLY_USE_CPU = bool(int(os.getenv('AOTRITON_TORCH_ONLY_USE_CPU', default='0')))
if AOTRITON_TORCH_ONLY_USE_CPU:
    from pyaotriton import HipMemory, hipDeviceSynchronize
else:
    # Let user import HipMemory unconditionally but its usage should be guarded with AOTRITON_TORCH_ONLY_USE_CPU
    HipMemory = None

def cast_dtype(dtype):
    assert not dtype.is_complex
    bits = dtype.itemsize * 8
    if dtype.is_floating_point:
        maintype = 'Float' if 'bfloat' not in str(dtype) else 'BFloat'
    else:
        maintype = 'Int' if 'uint' not in str(dtype) else 'UInt'
    typename = f'k{maintype}{bits}'
    return getattr(DType, typename)

def _do_mk_aotensor(q, if_empty_then_like=None, force_data_ptr=None):
    rank = len(q.shape) if q is not None else len(if_empty_then_like.shape)
    def lazy_data_ptr():
        return q.data_ptr() if force_data_ptr is None else force_data_ptr
    if q is not None and len(q.shape) == 1 and q.numel() in [0, 1]:
        if PASS_PHILOX_AS_TENSOR:
            return T0(lazy_data_ptr(), cast_dtype(q.dtype))
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
    return klass(lazy_data_ptr(), tuple(q.size()), q.stride(), cast_dtype(q.dtype))

if not AOTRITON_TORCH_ONLY_USE_CPU:
    def mk_aotensor(q, if_empty_then_like=None):
        return _do_mk_aotensor(q, if_empty_then_like=if_empty_then_like), q
else:
    def mk_aotensor(q, if_empty_then_like=None):
        if q is None or q.device.type != 'cpu':
            return _do_mk_aotensor(q, if_empty_then_like=if_empty_then_like), q
        devm = HipMemory()
        nbytes = q.untyped_storage().nbytes()
        devm.alloc(nbytes)
        devm.load_from_host(q.data_ptr(), nbytes)
        qview = _do_mk_aotensor(q,
                                if_empty_then_like=if_empty_then_like,
                                force_data_ptr=devm.get_pointer())
        return qview, devm

    def _torch_cpu_only_copy_back(cputensors, devms):
        hipDeviceSynchronize()
        for cput, devm in zip(cputensors, devms):
            if cput is None or devm is None:
                continue
            nbytes = cput.untyped_storage().nbytes()
            devm.store_to_host(cput.data_ptr(), nbytes)
        hipDeviceSynchronize()

def attn_fwd(q, k, v, b, sm_scale, M, o,
             dropout_p, philox_seed, philox_offset1, philox_offset2,
             philox_seed_output, philox_offset_output,
             encoded_softmax, is_causal,
             extargs=None):
    extargs = FwdExtraArguments() if extargs is None else extargs
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    Mview, Mdevm = mk_aotensor(M)
    oview, odevm = mk_aotensor(o)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    seedoutview, seedoutdevm = mk_aotensor(philox_seed_output)
    offsetoutview, offsetoutdevm = mk_aotensor(philox_offset_output)
    esmview, esmdevm = mk_aotensor(encoded_softmax, if_empty_then_like=q)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        hipDeviceSynchronize()
    err = fa_forward(qview,
                     kview,
                     vview,
                     bview,
                     float(sm_scale),
                     Mview,
                     oview,
                     float(dropout_p),
                     seedview,
                     offset1view,
                     philox_offset2,
                     seedoutview,
                     offsetoutview,
                     esmview,
                     is_causal,
                     Stream(),
                     extargs)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        _torch_cpu_only_copy_back([M, o, philox_seed_output, philox_offset_output, encoded_softmax],
                                  [Mdevm, odevm, seedoutdevm, offsetoutdevm, esmdevm])
    # print(f'{err=}')
    return err

def attn_bwd(q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, L, delta,
             dropout_p, philox_seed, philox_offset1, philox_offset2, is_causal, extargs=None):
    extargs = BwdExtraArguments() if extargs is None else extargs
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    oview, odevm = mk_aotensor(o)
    doutview, doutdevm = mk_aotensor(dout)
    dqview, dqdevm = mk_aotensor(dq)
    dkview, dkdevm = mk_aotensor(dk)
    dvview, dvdevm = mk_aotensor(dv)
    dbview, dbdevm = mk_aotensor(db, if_empty_then_like=q)
    Lview, Ldevm = mk_aotensor(L)
    deltaview, deltadevm = mk_aotensor(delta)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        hipDeviceSynchronize()
    # print(f'{b=}')
    err = fa_backward(qview,
                      kview,
                      vview,
                      bview,
                      float(sm_scale),
                      oview,
                      doutview,
                      dqview,
                      dkview,
                      dvview,
                      dbview,
                      Lview,
                      deltaview,
                      float(dropout_p),
                      seedview,
                      offset1view,
                      philox_offset2,
                      is_causal,
                      Stream(),
                      extargs)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        _torch_cpu_only_copy_back([dq, dk, dv, db, delta],
                                  [dqdevm, dkdevm, dvdevm, dbdevm, deltadevm])
    # print(f'{err=}')
    return err

def debug_fill_dropout_rng(R, philox_seed, philox_offset):
    Rview, Rdevm = mk_aotensor(R)
    err = fa_debug_fill_dropout_rng(Rview,
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
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    cuqview, cuqdevm = mk_aotensor(cu_seqlens_q),
    cukview, cukdevm = mk_aotensor(cu_seqlens_k),
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    Mview, Mdevm = mk_aotensor(M)
    oview, odevm = mk_aotensor(o)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    seedoutview, seedoutdevm = mk_aotensor(philox_seed_output)
    offsetoutview, offsetoutdevm = mk_aotensor(philox_offset_output)
    esmview, esmdevm = mk_aotensor(encoded_softmax, if_empty_then_like=q)
    err = fa_forward_compact_varlen(qview,
                                    kview,
                                    vview,
                                    cuqview,
                                    cukview,
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    bview,
                                    float(sm_scale),
                                    Mview,
                                    oview,
                                    float(dropout_p),
                                    seedview,
                                    offset1view,
                                    philox_offset2,
                                    seedoutview,
                                    offsetoutview,
                                    esmview,
                                    is_causal,
                                    Stream())
    # print(f'{err=}')
    return err

def attn_bwd_compact_varlen(q, k, v,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        b, sm_scale, o, dout, dq, dk, dv, db, L, delta,
        dropout_p, philox_seed, philox_offset1, philox_offset2, is_causal):
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    cuqview, cuqdevm = mk_aotensor(cu_seqlens_q),
    cukview, cukdevm = mk_aotensor(cu_seqlens_k),
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    oview, odevm = mk_aotensor(o)
    doutview, doutdevm = mk_aotensor(dout)
    dqview, dqdevm = mk_aotensor(dq)
    dkview, dkdevm = mk_aotensor(dk)
    dvview, dvdevm = mk_aotensor(dv)
    dbview, dbdevm = mk_aotensor(db, if_empty_then_like=q)
    Lview, Ldevm = mk_aotensor(L)
    deltaview, deltadevm = mk_aotensor(delta)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    # print(f'{b=}')
    err = fa_backward_compact_varlen(qview,
                                     kview,
                                     vview,
                                     cuqview,
                                     cukview,
                                     max_seqlen_q,
                                     max_seqlen_k,
                                     bview,
                                     float(sm_scale),
                                     oview,
                                     doutview,
                                     dqview,
                                     dkview,
                                     dvview,
                                     dbview,
                                     Lview,
                                     deltaview,
                                     float(dropout_p),
                                     seedview,
                                     offset1view,
                                     philox_offset2,
                                     is_causal,
                                     Stream())
    # print(f'{err=}')
    return err
