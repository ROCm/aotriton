# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, astuple
from argparse import Namespace
from ..kftdesc import KernelForTuneDescription as KFTDesc
from .flash import (
    FlashEntry,
    FlashInputMetadata,
    FlashKernelSelector,
)
from .reference import (
    SdpaReference,
    SdpaBidiInputs,
    SdpaGoldenOutputs,
)
from pyaotriton.v2 import CppTuneSpecialKernelIndex
from pyaotriton.v2.flash import (
    attn_fwd as fa_forward,
    attn_fwd_compact_varlen as fa_forward_compact_varlen,
    debug_simulate_encoded_softmax as fa_debug_simulate_encoded_softmax,
    attn_bwd as fa_backward,
    FwdExtraArguments,
    BwdExtraArguments,
    FusedBwdExtraArguments,
)
from ..gpu_utils import (
    target_fudge_factor,
    mk_aotensor,
    create_aotensor_like,
    zero_devm,
    translate_causal,
    Stream,
)

class CpptuneAccessor(object):
    FACTORY = None
    MEMBER = None

    def __init__(self):
        self._cpptune = self.FACTORY()

    @property
    def leaf(self):
        return getattr(self._cpptune, self.MEMBER)

    @property
    def force_kernel_index(self):
        return self.leaf.force_kernel_index

    @force_kernel_index.setter
    def force_kernel_index(self, ki):
        self.leaf.force_kernel_index = ki

    @property
    def total_number_of_kernels(self):
        return self.leaf.total_number_of_kernels

    @property
    def selected_kernel_copts(self):
        return self.leaf.selected_kernel_copts

    @property
    def selected_kernel_psels(self):
        return self.leaf.selected_kernel_psels

    @property
    def capi_object(self):
        return self._cpptune

    @property
    def peek_kernel_numbers(self):
        return self.leaf.peek_kernel_numbers

    @peek_kernel_numbers.setter
    def peek_kernel_numbers(self, value: bool):
        self.leaf.peek_kernel_numbers = value

class BwdExtraArgumentsCommon(CpptuneAccessor):
    FACTORY = BwdExtraArguments

class BwdExtraArgumentsDkDv(BwdExtraArgumentsCommon):
    MEMBER = 'dkdv'
    def __init__(self):
        super().__init__()
        self._cpptune.dqdb.force_kernel_index = CppTuneSpecialKernelIndex.kSkipGPUCall

class BwdExtraArgumentsDqDb(BwdExtraArgumentsCommon):
    MEMBER = 'dqdb'
    def __init__(self):
        super().__init__()
        self._cpptune.dkdv.force_kernel_index = CppTuneSpecialKernelIndex.kSkipGPUCall

class SdpaCommon(SdpaReference):
    EXT_CLASS = FwdExtraArguments

    def create_extargs(self, *, force_kernel_index=None, peek_kernel_numbers=None):
        ext = self.EXT_CLASS()
        if force_kernel_index is not None:
            ext.force_kernel_index = force_kernel_index
        if peek_kernel_numbers is not None:
            ext.peek_kernel_numbers = peek_kernel_numbers
        return ext

class attn_fwd(SdpaCommon):
    EXT_CLASS = FwdExtraArguments

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
        # VI API only has causal_type = True/False
        if inputs.window_sizes:
            view.causal_type = True
        else:
            view.causal_type = False
        view.stream = Stream()
        return im, view, devm

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.logsumexp.fill_(float("nan"))
        devm.out.fill_(float("nan"))

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        if view.atomic:
            zero_devm(devm.atomic)
        err = fa_forward(view.q,
                         view.k,
                         view.v,
                         view.b,
                         view.sm_scale,
                         view.logsumexp,
                         view.out,
                         im.dropout_p,
                         view.seed,
                         view.offset1,
                         view.offset2,
                         view.seedout,
                         view.offsetout,
                         view.esm,
                         view.causal_type,
                         view.atomic,
                         view.stream,
                         extargs)
        return (devm.out, devm.logsumexp)

    def compare(self, outputs, refs: SdpaGoldenOutputs):
        out, logsumexp = outputs
        return {"out": target_fudge_factor(out, refs.out)}

class bwd_kernel_dk_dv(SdpaCommon):
    EXT_CLASS = BwdExtraArgumentsDkDv

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
        view.delta, devm.delta = create_aotensor_like(inputs.logsumexp)
        view.dq, devm.dq = create_aotensor_like(inputs.q)
        view.dk, devm.dk = create_aotensor_like(inputs.k)
        view.dv, devm.dv = create_aotensor_like(inputs.v)
        view.db, devm.db = create_aotensor_like(inputs.b, if_none_then_like=inputs.q)
        view.seedout, devm.seedout = mk_aotensor(inputs.seedout)
        view.offset1, devm.offset1 = mk_aotensor(inputs.offsetout)
        view.offset2 = 0
        if inputs.window_sizes:
            view.causal_type = True
        else:
            view.causal_type = False
        view.stream = Stream()
        return im, view, devm

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.dk.fill_(float("nan"))
        devm.dv.fill_(float("nan"))

    def _direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        err = fa_backward(view.q,
                          view.k,
                          view.v,
                          view.b,
                          view.sm_scale,
                          view.out,
                          view.dout,
                          view.dq,
                          view.dk,
                          view.dv,
                          view.db,
                          view.logsumexp,
                          view.delta,
                          im.dropout_p,
                          view.seedout,
                          view.offset1,
                          view.offset2,
                          view.causal_type,
                          view.stream,
                          extargs.capi_object)
        return err

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        self._direct_call(direct_inputs, extargs)
        # print(f'{devm=}')
        return (devm.dk, devm.dv)

    def compare(self, outputs, refs: SdpaGoldenOutputs):
        dk, dv = outputs
        # print(f'dk.data_ptr = {dk.data_ptr():x}')
        # print(f'{dk=}')
        # print(f'{refs.dk[0]=}')
        return { "dk": target_fudge_factor(dk, refs.dk), "dv": target_fudge_factor(dv, refs.dv) }

class bwd_kernel_dq(SdpaCommon):
    EXT_CLASS = BwdExtraArgumentsDqDb

    prepare_directs = bwd_kernel_dk_dv.prepare_directs

    def fill_nan_to_outputs(self, direct_inputs):
        im, view, devm = direct_inputs
        devm.dq.fill_(float("nan"))
        if devm.db is not None:
            devm.db.fill_(float("nan"))

    def direct_call(self, direct_inputs, extargs):
        im, view, devm = direct_inputs
        bwd_kernel_dk_dv._direct_call(self, direct_inputs, extargs)
        return (devm.dq, devm.db)

    def compare(self, outputs, refs: SdpaGoldenOutputs):
        dq, db = outputs
        d = {}
        d["dq"] = target_fudge_factor(dq, refs.dq)
        d["db"] = target_fudge_factor(db, refs.db)
        return d

class bwd_kernel_fuse(SdpaCommon):
    EXT_CLASS = FusedBwdExtraArguments
