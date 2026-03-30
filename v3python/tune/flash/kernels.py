# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, astuple
from argparse import Namespace
from ..kftdesc import KernelForTuneDescription as KFTDesc
from .module import (
    FlashEntry,
    FlashInputMetadata,
)
from .reference import (
    SdpaReference,
    SdpaBidiInputs,
    SdpaGoldenOutputs,
)
from pyaotriton.v3 import KernelControl
from pyaotriton.v3.flash import (
    attn_fwd as fa_forward_op,
    attn_fwd_params as fa_forward_op_params,
    attn_bwd as fa_backward_op,
    attn_bwd_params as fa_backward_op_params,
    attn_options,
)
from ..gpu_utils import (
    target_fudge_factor,
    mk_aotensor,
    create_aotensor_like,
    zero_devm,
    translate_causal,
    Stream,
    cast_dtype,
)
from pyaotriton import T2, T4

NAN = float("nan")

# Triton kernel does not use dq_acc tensor so it is safe to create an empty one
def eager_null_dq_acc(dq):
    from pyaotriton import lazy_tensor
    dq_view = T4(0, tuple(dq.size()), dq.stride(), cast_dtype(dq.dtype))
    return lazy_tensor.eager_null_dq_acc(dq_view)

def eager_delta(L):
    from pyaotriton import lazy_tensor
    L_view = T2(L.data_ptr(), tuple(L.size()), L.stride(), cast_dtype(L.dtype))
    return lazy_tensor.eager_delta(L_view)

class AttnOptionsWrapper:
    C_CLASS = attn_options

    def __init__(self, backend: int, slot: int):
        self._c = self.C_CLASS()
        self._backend = backend
        self._c.force_backend_index = self._backend
        self._slot = slot
        self.ignore_all_kernels()

    @property
    def c_object(self):
        return self._c

    '''
    |            | probe=True                    | probe=False          |
    | ---------- | ----------------------------- | -------------------- |
    | hsaco=int  | Skip hsaco, return psel/copt  | Run selected hsaco   |
    | hsaco=None | Return total number of hsacos | Run auto tune kernel |
    '''
    def set_hsaco(self, hsaco: int|None = None, probe: bool = False):
        c = self._c
        slot = self._slot
        ctrl = KernelControl.Default
        if hsaco is not None:
            ctrl = ctrl | KernelControl.Manual
            c.kernel_fine_control[slot].hsaco_index = hsaco
        if probe:
            ctrl = ctrl | KernelControl.Query | KernelControl.Skip
        c.kernel_fine_control[slot].control_bits = ctrl

    '''
    Unlike set_hsaco, None means "don't change"
    '''
    def update_hsaco(self, hsaco: int|None = None, probe: bool|None = None):
        c = self._c
        slot = self._slot
        kfc = c.kernel_fine_control[slot]
        current_hsaco = kfc.hsaco_index
        current_probe = kfc.control_bits & KernelControl.Query
        update_hsaco = current_hsaco if hsaco is None else hsaco
        update_probe = current_probe if probe is None else probe
        self.set_hsaco(update_hsaco, update_probe)

    def ignore_all_kernels(self):
        c = self._c
        for slot in range(int(c.KernelSlot.MaxKernels)):
            c.kernel_fine_control[slot].control_bits = KernelControl.Ignore

    @property
    def selected_kernel_total_hsacos(self):
        return self._c.kernel_fine_control[self._slot].total_hsacos

    @property
    def selected_hsaco_psels(self):
        return self._c.kernel_fine_control[self._slot].kernel_psels

    @property
    def selected_hsaco_copts(self):
        return self._c.kernel_fine_control[self._slot].kernel_copts


# Common code for All SDPA kernels
#
# PRE-CONDITION
#     * KernelDescription.NAME == attn_options.KernelSlot.<name> == class_name(SdpaCommon)
class SdpaCommon(SdpaReference):
    EXT_CLASS = AttnOptionsWrapper
    BACKEND_INDEX = None  # Must define in subclass

    def create_extargs(self, *, hsaco_index=None, probe=False):
        ext = self.EXT_CLASS(self.BACKEND_INDEX, self.KERNEL_SLOT)
        ext.set_hsaco(hsaco=hsaco_index, probe=probe)
        return ext

    @property
    def KERNEL_SLOT(self):
        return int(getattr(self.EXT_CLASS.C_CLASS, self.__class__.__name__))

    OUTPUT_TNAMES = None

    def compare(self, outputs, refs: SdpaGoldenOutputs):
        d = {}
        for tn, t in zip(self.OUTPUT_TNAMES, outputs):
            d[tn] = target_fudge_factor(t, getattr(refs, tn))
        return d

class attn_fwd(SdpaCommon):
    EXT_CLASS = AttnOptionsWrapper
    BACKEND_INDEX = 0

    def prepare_directs(self, im: FlashInputMetadata, inputs: SdpaBidiInputs):
        view = Namespace()
        devm = Namespace()
        view.q, devm.q = mk_aotensor(inputs.q)
        view.k, devm.k = mk_aotensor(inputs.k)
        view.v, devm.v = mk_aotensor(inputs.v)
        view.b, devm.b = mk_aotensor(inputs.b, if_empty_then_like=inputs.q)
        view.sm_scale = inputs.sm_scale
        view.logsumexp, devm.logsumexp = create_aotensor_like(inputs.logsumexp)
        view.out, devm.out = mk_aotensor(inputs.out)
        view.seed, devm.seed = mk_aotensor(inputs.seed)
        view.offset1, devm.offset1 = mk_aotensor(inputs.offset1)
        view.offset2 = inputs.offset2
        view.seedout, devm.seedout = mk_aotensor(inputs.seedout)
        view.offsetout, devm.offsetout = mk_aotensor(inputs.offsetout)
        view.esm, devm.esm = mk_aotensor(inputs.encoded_softmax, if_empty_then_like=inputs.q)
        view.atomic, devm.atomic = mk_aotensor(inputs.atomic)
        # V3 API uses causal_type and window values
        if inputs.window_sizes:
            view.causal_type, view.window_left, view.window_right = translate_causal(True, v3_api=True)
        else:
            view.causal_type, view.window_left, view.window_right = translate_causal(False, v3_api=True)
        view.stream = Stream()
        return im, view, devm

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.logsumexp.fill_(NAN)
        devm.out.fill_(NAN)

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        if view.atomic:
            zero_devm(devm.atomic)
        params = fa_forward_op_params()
        params.Q = view.q
        params.K = view.k
        params.V = view.v
        params.B = view.b
        params.Sm_scale = float(view.sm_scale)
        params.L = view.logsumexp
        params.Out = view.out
        params.dropout_p = float(im.dropout_p)
        params.philox_seed_ptr = view.seed
        params.philox_offset1 = view.offset1
        params.philox_offset2 = view.offset2
        params.philox_seed_output = view.seedout
        params.philox_offset_output = view.offsetout
        params.encoded_softmax = view.esm
        params.persistent_atomic_counter = view.atomic
        params.causal_type = view.causal_type
        params.window_left = view.window_left
        params.window_right = view.window_right
        params.varlen_type = 0
        err = fa_forward_op(params,
                            fa_forward_op_params.kVersion,
                            view.stream,
                            extargs.c_object)
        return (devm.out, devm.logsumexp)

    OUTPUT_TNAMES = ["out"]

class bwd_kernel_dk_dv(SdpaCommon):
    EXT_CLASS = AttnOptionsWrapper
    BACKEND_INDEX = 0

    def prepare_directs(self, im: FlashInputMetadata, inputs: SdpaBidiInputs):
        view = Namespace()
        devm = Namespace()
        view.q, devm.q = mk_aotensor(inputs.q)
        view.k, devm.k = mk_aotensor(inputs.k)
        view.v, devm.v = mk_aotensor(inputs.v)
        view.b, devm.b = mk_aotensor(inputs.b, if_empty_then_like=inputs.q)
        view.sm_scale = inputs.sm_scale
        view.out, devm.out = mk_aotensor(inputs.out)
        view.dout, devm.dout = mk_aotensor(inputs.dout)
        # bwd uses LSE as it is
        view.logsumexp, devm.logsumexp = mk_aotensor(inputs.logsumexp)
        # V3 API: delta is an eager lazy tensor (no allocation overhead)
        view.delta, devm.delta = mk_aotensor(inputs.delta)
        view.delta = eager_delta(devm.delta)
        # V3 API: dq_acc is a lazy tensor
        view.dq, devm.dq = create_aotensor_like(inputs.q)
        view.dq_acc = eager_null_dq_acc(devm.dq)
        view.dk, devm.dk = create_aotensor_like(inputs.k)
        view.dv, devm.dv = create_aotensor_like(inputs.v)
        view.db, devm.db = create_aotensor_like(inputs.b, if_none_then_like=inputs.q)
        view.seedout, devm.seedout = mk_aotensor(inputs.seedout)
        view.offset1, devm.offset1 = mk_aotensor(inputs.offsetout)
        view.offset2 = 0
        # V3 API uses causal_type and window values
        if inputs.window_sizes:
            view.causal_type, view.window_left, view.window_right = translate_causal(True, v3_api=True)
        else:
            view.causal_type, view.window_left, view.window_right = translate_causal(False, v3_api=True)
        view.stream = Stream()
        return im, view, devm

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.dk.fill_(NAN)
        devm.dv.fill_(NAN)

    def _direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        params = fa_backward_op_params()
        params.Q = view.q
        params.K = view.k
        params.V = view.v
        params.B = view.b
        params.Sm_scale = float(view.sm_scale)
        params.Out = view.out
        params.DO = view.dout
        params.DK = view.dk
        params.DV = view.dv
        params.DQ = view.dq
        params.DB = view.db
        params.DQ_ACC = view.dq_acc
        params.L = view.logsumexp
        params.D = view.delta
        params.dropout_p = float(im.dropout_p)
        params.philox_seed_ptr = view.seedout
        params.philox_offset1 = view.offset1
        params.philox_offset2 = view.offset2
        params.causal_type = view.causal_type
        params.window_left = view.window_left
        params.window_right = view.window_right
        params.varlen_type = 0
        err = fa_backward_op(params,
                             fa_backward_op_params.kVersion,
                             view.stream,
                             extargs.c_object)
        return err

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        self._direct_call(direct_inputs, extargs)
        # print(f'{devm=}')
        return (devm.dk, devm.dv)

    OUTPUT_TNAMES = ["dk", "dv"]

class bwd_kernel_dq(SdpaCommon):
    EXT_CLASS = AttnOptionsWrapper
    BACKEND_INDEX = 0

    prepare_directs = bwd_kernel_dk_dv.prepare_directs

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.dq.fill_(NAN)
        if devm.db is not None:
            devm.db.fill_(NAN)

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        params = fa_backward_op_params()
        params.Q = view.q
        params.K = view.k
        params.V = view.v
        params.B = view.b
        params.Sm_scale = float(view.sm_scale)
        params.Out = view.out
        params.DO = view.dout
        params.DK = view.dk
        params.DV = view.dv
        params.DQ = view.dq
        params.DB = view.db
        params.DQ_ACC = view.dq_acc
        params.L = view.logsumexp
        params.D = view.delta
        params.dropout_p = float(im.dropout_p)
        params.philox_seed_ptr = view.seedout
        params.philox_offset1 = view.offset1
        params.philox_offset2 = view.offset2
        params.causal_type = view.causal_type
        params.window_left = view.window_left
        params.window_right = view.window_right
        params.varlen_type = 0
        err = fa_backward_op(params,
                             fa_backward_op_params.kVersion,
                             view.stream,
                             extargs.c_object)
        return (devm.dq, devm.db)

    OUTPUT_TNAMES = ["dq", "db"]

class bwd_kernel_fuse(SdpaCommon):
    EXT_CLASS = AttnOptionsWrapper
    BACKEND_INDEX = 1

    def prepare_directs(self, im: FlashInputMetadata, inputs: SdpaBidiInputs):
        view = Namespace()
        devm = Namespace()
        view.q, devm.q = mk_aotensor(inputs.q)
        view.k, devm.k = mk_aotensor(inputs.k)
        view.v, devm.v = mk_aotensor(inputs.v)
        view.b, devm.b = mk_aotensor(inputs.b, if_empty_then_like=inputs.q)
        view.sm_scale = inputs.sm_scale
        view.out, devm.out = mk_aotensor(inputs.out)
        view.dout, devm.dout = mk_aotensor(inputs.dout)
        # bwd uses LSE as it is
        view.logsumexp, devm.logsumexp = mk_aotensor(inputs.logsumexp)
        # V3 API: delta is an eager lazy tensor (no allocation overhead)
        view.delta, devm.delta = mk_aotensor(inputs.delta)
        view.delta = eager_delta(devm.delta)
        # V3 API: dq_acc is a lazy tensor
        view.dq, devm.dq = create_aotensor_like(inputs.q)
        view.dq_acc = eager_null_dq_acc(devm.dq)
        view.dk, devm.dk = create_aotensor_like(inputs.k)
        view.dv, devm.dv = create_aotensor_like(inputs.v)
        view.db, devm.db = create_aotensor_like(inputs.b, if_none_then_like=inputs.q)
        view.seedout, devm.seedout = mk_aotensor(inputs.seedout)
        view.offset1, devm.offset1 = mk_aotensor(inputs.offsetout)
        view.offset2 = 0
        # V3 API uses causal_type and window values
        if inputs.window_sizes:
            view.causal_type, view.window_left, view.window_right = translate_causal(True, v3_api=True)
        else:
            view.causal_type, view.window_left, view.window_right = translate_causal(False, v3_api=True)
        view.stream = Stream()
        return im, view, devm

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.dk.fill_(NAN)
        devm.dv.fill_(NAN)
        devm.dq.fill_(NAN)
        if devm.db is not None:
            devm.db.fill_(NAN)

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        params = fa_backward_op_params()
        params.Q = view.q
        params.K = view.k
        params.V = view.v
        params.B = view.b
        params.Sm_scale = float(view.sm_scale)
        params.Out = view.out
        params.DO = view.dout
        params.DK = view.dk
        params.DV = view.dv
        params.DQ = view.dq
        params.DB = view.db
        params.DQ_ACC = view.dq_acc
        params.L = view.logsumexp
        params.D = view.delta
        params.dropout_p = float(im.dropout_p)
        params.philox_seed_ptr = view.seedout
        params.philox_offset1 = view.offset1
        params.philox_offset2 = view.offset2
        params.causal_type = view.causal_type
        params.window_left = view.window_left
        params.window_right = view.window_right
        params.varlen_type = 0
        err = fa_backward_op(params,
                             fa_backward_op_params.kVersion,
                             view.stream,
                             extargs.c_object)
        return (devm.dk, devm.dv, devm.dq, devm.db)

    OUTPUT_TNAMES = ["dk", "dv", "dq", "db"]
