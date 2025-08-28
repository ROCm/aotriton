# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
IGNORE_BACKWARD_IMPORT = bool(int(os.getenv('IGNORE_BACKWARD_IMPORT', default='0')))

from pyaotriton.v2.flash import (
    attn_fwd as fa_forward,
    attn_fwd_compact_varlen as fa_forward_compact_varlen,
    # debug_fill_dropout_rng as fa_debug_fill_dropout_rng,
    debug_simulate_encoded_softmax as fa_debug_simulate_encoded_softmax,
    FwdExtraArguments,
)
from pyaotriton.v3.flash import (
    attn_fwd as fa_forward_op,
    attn_fwd_params as fa_forward_op_params,
    attn_options,
)
if not IGNORE_BACKWARD_IMPORT:
    from pyaotriton.v2.flash import (
        attn_bwd as fa_backward,
        attn_bwd_fused as fa_backward_fused,
        attn_bwd_compact_varlen as fa_backward_compact_varlen,
        BwdExtraArguments,
        FusedBwdExtraArguments,
    )
    from pyaotriton.v3.flash import (
        attn_bwd as fa_backward_op,
        attn_bwd_params as fa_backward_op_params,
        aiter_bwd as fa_aiter_bwd,
    )

# Note: we don't use Enum class because accessing the integer requires using
#       `.value` property, which makes the code verbose.
class CausalType:
    NONE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2
    WINDOWED = 3

class WindowValue:
    NONE = 0
    TOP_LEFT_ALIGNED = -2147483647       # 0x80000001. Special value for varlen
    BOTTOM_RIGHT_ALIGNED = -2147483646   # 0x80000002. Special value for varlen

def translate_causal(causal, v3_api):
    window_left, window_right = 0, 0
    if isinstance(causal, tuple):
        assert v3_api, 'Only V3_API supports windowed attention (causal = tuple([window_left, window_right]))'
        window_left, window_right = causal
        causal_type = CausalType.WINDOWED
    elif isinstance(causal, bool):
        # causal_type = CausalType.TOP_LEFT if causal else CausalType.NONE
        causal_type = CausalType.WINDOWED if causal else CausalType.NONE
        if causal:
            window_left = WindowValue.TOP_LEFT_ALIGNED
            window_right = WindowValue.TOP_LEFT_ALIGNED
    else:
        assert causal in [CausalType.NONE, CausalType.TOP_LEFT, CausalType.BOTTOM_RIGHT]
        assert v3_api, 'CausalType.TOP_LEFT/BOTTOM_RIGHT variant is supported thru windowed attention, which requires V3 API'
        if causal == CausalType.TOP_LEFT:
            causal_type = CausalType.WINDOWED
            window_left = WindowValue.TOP_LEFT_ALIGNED
            window_right = WindowValue.TOP_LEFT_ALIGNED
        elif causal == CausalType.BOTTOM_RIGHT:
            causal_type = CausalType.WINDOWED
            window_left = WindowValue.BOTTOM_RIGHT_ALIGNED
            window_right = WindowValue.BOTTOM_RIGHT_ALIGNED
        else:
            causal_type = causal
    return causal_type, window_left, window_right

from pyaotriton import T1, T2, T4, DType, Stream, hipError_t, get_name_suffix, hipGetLastError
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
             encoded_softmax, causal, atomic,
             extargs=None, call_operator=False):
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
    atomicview, atomicdevm = mk_aotensor(atomic)
    causal_type, window_left, window_right = translate_causal(causal, v3_api=call_operator)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        hipDeviceSynchronize()
    if not call_operator:
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
                         causal_type,
                         atomicview,
                         Stream(),
                         extargs)
    else:
        params = fa_forward_op_params()
        params.Q = qview
        params.K = kview
        params.V = vview
        params.B = bview
        params.Sm_scale = float(sm_scale)
        params.L = Mview
        params.Out = oview
        # params.cu_seqlens_q
        # params.cu_seqlens_k
        # params.Max_seqlen_q
        # params.Max_seqlen_k
        params.dropout_p = float(dropout_p)
        params.philox_seed_ptr = seedview
        params.philox_offset1 = offset1view
        params.philox_offset2 = philox_offset2
        params.philox_seed_output = seedoutview
        params.philox_offset_output = offsetoutview
        params.encoded_softmax = esmview
        params.persistent_atomic_counter = atomicview
        params.causal_type = causal_type
        params.window_left = window_left
        params.window_right = window_right
        params.varlen_type = 0
        err = fa_forward_op(params,
                            fa_forward_op_params.kVersion,
                            Stream(),
                            attn_options()
                            )
    if AOTRITON_TORCH_ONLY_USE_CPU:
        _torch_cpu_only_copy_back([M, o, philox_seed_output, philox_offset_output, encoded_softmax],
                                  [Mdevm, odevm, seedoutdevm, offsetoutdevm, esmdevm])
    # print(f'{err=}')
    return err

def attn_bwd(q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, dq_acc, L, delta,
             dropout_p, philox_seed, philox_offset1, philox_offset2, causal,
             extargs=None, call_operator=False):
    if call_operator:
        extargs = attn_options() if extargs is None else extargs
    else:
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
    if call_operator:
        deltaview = delta
    else:
        deltaview, deltadevm = mk_aotensor(delta)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        hipDeviceSynchronize()
    causal_type, window_left, window_right = translate_causal(causal, v3_api=call_operator)
    if not call_operator:
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
                          causal,
                          Stream(),
                          extargs)
    else:
        params = fa_backward_op_params()
        params.Q = qview;
        params.K = kview;
        params.V = vview;
        params.B = bview;
        params.Sm_scale = float(sm_scale);
        params.Out = oview;
        params.DO = doutview;
        params.DK = dkview;
        params.DV = dvview;
        params.DQ = dqview;
        params.DB = dbview;
        params.DQ_ACC = dq_acc;
        params.L = Lview;
        params.D = deltaview;
        # params.cu_seqlens_q
        # params.cu_seqlens_k
        # params.Max_seqlen_q
        # params.Max_seqlen_k
        params.dropout_p = float(dropout_p);
        params.philox_seed_ptr = seedview;
        params.philox_offset1 = offset1view;
        params.philox_offset2 = philox_offset2;
        params.causal_type = causal_type
        params.window_left = window_left
        params.window_right = window_right
        params.varlen_type = 0
        err = fa_backward_op(params,
                             fa_backward_op_params.kVersion,
                             Stream(),
                             extargs)
    if AOTRITON_TORCH_ONLY_USE_CPU:  # FIXME: V3+CPU
        _torch_cpu_only_copy_back([dq, dk, dv, db, delta],
                                  [dqdevm, dkdevm, dvdevm, dbdevm, deltadevm])
    return err

def attn_bwd_fused(q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, L,
             dropout_p, philox_seed, philox_offset1, philox_offset2, causal, extargs=None):
    extargs = FusedBwdExtraArguments() if extargs is None else extargs
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
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        hipDeviceSynchronize()
    causal_type, window_left, window_right = translate_causal(causal, v3_api=False)
    # print(f'{b=}')
    err = fa_backward_fused(qview,
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
                            float(dropout_p),
                            seedview,
                            offset1view,
                            philox_offset2,
                            causal_type,
                            Stream(),
                            extargs)
    if AOTRITON_TORCH_ONLY_USE_CPU:
        _torch_cpu_only_copy_back([dq, dk, dv, db],
                                  [dqdevm, dkdevm, dvdevm, dbdevm])
    # print(f'{err=}')
    return err

def attn_bwd_aiter(q, k, v, b, sm_scale, o, dout, dq, dk, dv, db, dq_acc, L, delta,
                   dropout_p, philox_seed, philox_offset1, philox_offset2, causal, extargs=None):
    extargs = BwdExtraArguments() if extargs is None else extargs
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    oview, odevm = mk_aotensor(o)
    doutview, doutdevm = mk_aotensor(dout)
    # dqaccview, dqaccdevm = mk_aotensor(dq_acc)
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
    # Keep using v3_api=False since we are not exposing Windowed Attention ATM
    causal_type, window_left, window_right = translate_causal(causal, v3_api=False)
    params = fa_backward_op_params()
    params.Q = qview
    params.K = kview
    params.V = vview
    params.B = bview
    params.Sm_scale = float(sm_scale)
    params.Out = oview
    params.DO = doutview
    params.DQ = dqview
    params.DK = dkview
    params.DV = dvview
    params.DQ_ACC = dq_acc
    params.DB = dbview
    params.L = Lview
    params.D = deltaview
    # params.cu_seqlens_q
    # params.cu_seqlens_k
    # params.Max_seqlen_q
    # params.Max_seqlen_k
    params.dropout_p = float(dropout_p)
    params.philox_seed_ptr = seedview
    params.philox_offset1 = offset1view
    params.philox_offset2 = philox_offset2
    params.causal_type = causal_type
    params.window_left = window_left
    params.window_right = window_right
    params.varlen_type = 0
    err = fa_aiter_bwd(params,
                       fa_backward_op_params.kVersion,
                       Stream(),
                       attn_options()
                       )
    if AOTRITON_TORCH_ONLY_USE_CPU:
        _torch_cpu_only_copy_back([dq_acc, dk, dv, db, delta],
                                  [dqaccdevm, dkdevm, dvdevm, dbdevm, deltadevm])
    return err

# def debug_fill_dropout_rng(R, philox_seed, philox_offset):
#     Rview, Rdevm = mk_aotensor(R)
#     err = fa_debug_fill_dropout_rng(Rview,
#                                     philox_seed,
#                                     philox_offset,
#                                     Stream())
#     # print(f'debug_fill_dropout_rng {err=}')
#     return err

def debug_simulate_encoded_softmax(R, dropout_p, philox_seed, philox_offset1, philox_offset2):
    Rview, Rdevm = mk_aotensor(R)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offsetview, offsetdevm = mk_aotensor(philox_offset1)
    err = fa_debug_simulate_encoded_softmax(Rview,
                                            dropout_p,
                                            seedview,
                                            offsetview,
                                            philox_offset2,
                                            Stream())
    return err

def attn_fwd_compact_varlen(q, k, v,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        b, sm_scale, M, o,
        dropout_p, philox_seed, philox_offset1, philox_offset2,
        philox_seed_output, philox_offset_output,
        encoded_softmax, causal, atomic, call_operator=False):
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    cuqview, cuqdevm = mk_aotensor(cu_seqlens_q)
    cukview, cukdevm = mk_aotensor(cu_seqlens_k)
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    Mview, Mdevm = mk_aotensor(M)
    oview, odevm = mk_aotensor(o)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    seedoutview, seedoutdevm = mk_aotensor(philox_seed_output)
    offsetoutview, offsetoutdevm = mk_aotensor(philox_offset_output)
    esmview, esmdevm = mk_aotensor(encoded_softmax, if_empty_then_like=q)
    atomicview, atomicdevm = mk_aotensor(atomic)
    causal_type, window_left, window_right = translate_causal(causal, v3_api=call_operator)
    if not call_operator:
        err = fa_forward_compact_varlen(qview,
                                        kview,
                                        vview,
                                        bview,
                                        cuqview,
                                        cukview,
                                        max_seqlen_q,
                                        max_seqlen_k,
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
                                        causal_type,
                                        atomicview,
                                        Stream())
    else:
        params = fa_forward_op_params()
        params.Q = qview
        params.K = kview
        params.V = vview
        params.B = bview
        params.Sm_scale = float(sm_scale)
        params.L = Mview
        params.Out = oview
        params.cu_seqlens_q = cuqview
        params.cu_seqlens_k = cukview
        params.Max_seqlen_q = max_seqlen_q
        params.Max_seqlen_k = max_seqlen_k
        params.dropout_p = float(dropout_p)
        params.philox_seed_ptr = seedview
        params.philox_offset1 = offset1view
        params.philox_offset2 = philox_offset2
        params.philox_seed_output = seedoutview
        params.philox_offset_output = offsetoutview
        params.encoded_softmax = esmview
        params.persistent_atomic_counter = atomicview
        params.causal_type = causal_type
        params.window_left = window_left
        params.window_right = window_right
        params.varlen_type = 1
        err = fa_forward_op(params,
                            fa_forward_op_params.kVersion,
                            Stream(),
                            attn_options()
                            )
    # print(f'{err=}')
    return err

def attn_bwd_compact_varlen(q, k, v,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        b, sm_scale, o, dout, dq, dk, dv, db, L, delta,
        dropout_p, philox_seed, philox_offset1, philox_offset2, causal, call_operator=False):
    qview, qdevm = mk_aotensor(q)
    kview, kdevm = mk_aotensor(k)
    vview, vdevm = mk_aotensor(v)
    cuqview, cuqdevm = mk_aotensor(cu_seqlens_q)
    cukview, cukdevm = mk_aotensor(cu_seqlens_k)
    bview, bdevm = mk_aotensor(b, if_empty_then_like=q)
    oview, odevm = mk_aotensor(o)
    doutview, doutdevm = mk_aotensor(dout)
    dqview, dqdevm = mk_aotensor(dq)
    dkview, dkdevm = mk_aotensor(dk)
    dvview, dvdevm = mk_aotensor(dv)
    dbview, dbdevm = mk_aotensor(db, if_empty_then_like=q)
    Lview, Ldevm = mk_aotensor(L)
    if call_operator:
        deltaview = delta
    else:
        deltaview, deltadevm = mk_aotensor(delta)
    seedview, seeddevm = mk_aotensor(philox_seed)
    offset1view, offset1devm = mk_aotensor(philox_offset1)
    causal_type, window_left, window_right = translate_causal(causal, v3_api=call_operator)
    # print(f'{b=}')
    if not call_operator:
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
                                         causal_type,
                                         Stream())
    else:
        params = fa_backward_op_params()
        params.Q = qview;
        params.K = kview;
        params.V = vview;
        params.B = bview;
        params.Sm_scale = float(sm_scale);
        params.Out = oview;
        params.DO = doutview;
        params.DK = dkview;
        params.DV = dvview;
        params.DQ = dqview;
        params.DB = dbview;
        params.L = Lview;
        params.D = deltaview;
        params.cu_seqlens_q = cuqview
        params.cu_seqlens_k = cukview
        params.Max_seqlen_q = max_seqlen_q
        params.Max_seqlen_k = max_seqlen_k
        params.dropout_p = float(dropout_p);
        params.philox_seed_ptr = seedview;
        params.philox_offset1 = offset1view;
        params.philox_offset2 = philox_offset2;
        params.causal_type = causal_type
        params.window_left = window_left
        params.window_right = window_right
        params.varlen_type = 1
        err = fa_backward_op(params,
                             fa_backward_op_params.kVersion,
                             Stream(),
                             attn_options()
                             )
    # print(f'{err=}')
    return err

def lazy_dq_acc(dq : 'torch.Tensor'):
    from pyaotriton import lazy_tensor
    dq_view = T4(dq.data_ptr(), tuple(dq.size()), dq.stride(), cast_dtype(dq.dtype))
    return lazy_tensor.dq_acc(dq_view, dq.device.index)

def lazy_delta(L : 'torch.Tensor'):
    from pyaotriton import lazy_tensor
    L_view = T2(L.data_ptr(), tuple(L.size()), L.stride(), cast_dtype(L.dtype))
    return lazy_tensor.delta(L_view, L.device.index)
