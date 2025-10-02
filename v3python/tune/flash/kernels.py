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
from pyaotriton.v2.flash import (
    attn_fwd as fa_forward,
    attn_fwd_compact_varlen as fa_forward_compact_varlen,
    debug_simulate_encoded_softmax as fa_debug_simulate_encoded_softmax,
    FwdExtraArguments,
)
from ..gpu_utils import (
    mk_aotensor,
    zero_devm,
    translate_causal,
    Stream,
)

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
        view.logsumexp, devm.logsumexp = mk_aotensor(inputs.logsumexp)
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

    def compare(self, outputs, refs: SdpaGoldenOutputs):
        raise NotImplemented('attn_fwd.compare')

class bwd_kernel_dk_dv(SdpaCommon):
    pass

class bwd_kernel_dq(SdpaCommon):
    pass

class bwd_kernel_fuse(SdpaCommon):
    pass
