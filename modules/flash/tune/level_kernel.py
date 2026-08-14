# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Kernel-level tuning strategy for flash (modular-tune.md §4.1/§4.3): selects
among HSACO variants of a single Triton kernel (attn_fwd / bwd_kernel_dk_dv /
bwd_kernel_dq / bwd_kernel_fuse), using KernelControl / kernel_fine_control.
Needs the *tuning*-instrumented pyaotriton library build (REQUIRES_TUNING_LIB
= True), as opposed to the plain testing build used by level_op.py.

IMPORTANT: this module must stay torch/pyaotriton-free AT MODULE SCOPE --
dispatch_tasks.py instantiates `TuneDesc()` (default level='kernel') for
*every* registered tuning module at CLI startup, purely to build argparse
subparsers from get_entry_choices(), on machines that may not have
torch/pyaotriton installed at all (see tdesc.py's module docstring). All
torch/pyaotriton imports below are therefore deferred into get_impl() /
enumerate_variants() / impl_desc(), never at module level or in __init__ --
mirroring the pre-unification split between flash/module.py (torch-free) and
flash/kernels.py (lazily imported only from Flash.get_impl()).
"""

from aotriton.tune.tdesc import TuningLevel

_KERNEL_DICT_CACHE = None


def _build_kernel_dict():
    """Lazily compose the tuning-lib kernel classes (KernelControl-based HSACO
    selection layered onto the plain SdpaCalls direct_call implementations).
    Only called once, from FlashKernelLevel.get_impl(); cached at module scope
    so every FlashKernelLevel instance shares one dict, same as the original
    Flash.KERNEL_DICT class attribute."""
    from pyaotriton.v3 import KernelControl
    from pyaotriton.v3.flash import attn_options
    from .calls import (
        SdpaCalls,
        attn_fwd as _attn_fwd,
        bwd_kernel_dk_dv as _bwd_kernel_dk_dv,
        bwd_kernel_dq as _bwd_kernel_dq,
        bwd_kernel_fuse as _bwd_kernel_fuse,
    )

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
        def set_hsaco(self, hsaco: int | None = None, probe: bool = False):
            c = self._c
            slot = self._slot
            ctrl = KernelControl.Default
            if hsaco is not None:
                ctrl = ctrl | KernelControl.Manual
                c.kernel_fine_control[slot].hsaco_index = hsaco
            if probe:
                ctrl = ctrl | KernelControl.Query | KernelControl.Skip
            c.kernel_fine_control[slot].control_bits = ctrl

        def disable_probing(self):
            """Switch from probe mode to run mode (clear Query/Skip bits, keep Manual/hsaco)."""
            self.update_hsaco(probe=False)

        '''
        Unlike set_hsaco, None means "don't change"
        '''
        def update_hsaco(self, hsaco: int | None = None, probe: bool | None = None):
            c = self._c
            slot = self._slot
            kfc = c.kernel_fine_control[slot]
            current_hsaco = kfc.hsaco_index if (kfc.control_bits & KernelControl.Manual) else None
            current_probe = bool(kfc.control_bits & KernelControl.Query)
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

        @classmethod
        def for_op_backend(cls, backend_index: int) -> 'AttnOptionsWrapper':
            '''Force a backend but leave all kernel slots at Default so the runtime
            looks up hsacos from the tuning DB normally (no Ignore/Manual bits).'''
            obj = cls.__new__(cls)
            obj._c = cls.C_CLASS()
            obj._backend = backend_index
            obj._slot = None
            obj._c.force_backend_index = backend_index
            return obj

    class SdpaOpts:
        """Mixin providing attn_options-based kernel selection for tuning."""
        EXT_CLASS = AttnOptionsWrapper
        BACKEND_INDEX = None  # Must define in subclass

        def create_extargs(self, *, which_impl=None, probe=False):
            hsaco_index = which_impl.impl_index if which_impl is not None else None
            ext = self.EXT_CLASS(self.BACKEND_INDEX, self.KERNEL_SLOT)
            ext.set_hsaco(hsaco=hsaco_index, probe=probe)
            return ext

        @property
        def KERNEL_SLOT(self):
            return int(getattr(self.EXT_CLASS.C_CLASS, self.__class__.__name__))

    class attn_fwd(SdpaOpts, _attn_fwd):
        BACKEND_INDEX = 0

    class bwd_kernel_dk_dv(SdpaOpts, _bwd_kernel_dk_dv):
        BACKEND_INDEX = 0

    class bwd_kernel_dq(SdpaOpts, _bwd_kernel_dq):
        BACKEND_INDEX = 0

    class bwd_kernel_fuse(SdpaOpts, _bwd_kernel_fuse):
        BACKEND_INDEX = 1

    return {
        'attn_fwd'          : attn_fwd(),
        'bwd_kernel_dk_dv'  : bwd_kernel_dk_dv(),
        'bwd_kernel_dq'     : bwd_kernel_dq(),
        'bwd_kernel_fuse'   : bwd_kernel_fuse(),
    }


class FlashKernelLevel(TuningLevel):
    NAME = 'kernel'
    REQUIRES_TUNING_LIB = True

    def list_impls(self, entry, arch: str | None = None) -> list[str]:
        if entry.hdim > 224:
            return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq']
        return ['attn_fwd', 'bwd_kernel_dk_dv', 'bwd_kernel_dq', 'bwd_kernel_fuse']

    def get_impl(self, name: str):
        global _KERNEL_DICT_CACHE
        if _KERNEL_DICT_CACHE is None:
            _KERNEL_DICT_CACHE = _build_kernel_dict()
        return _KERNEL_DICT_CACHE[name]

    def enumerate_variants(self, entry, im, which_impl: str, pt) -> list[dict]:
        import torch
        from dacite import from_dict
        from aotriton.tune.gpu_utils import device_ctx, default_device_string
        from aotriton.tune.utils import safeload, dacite_tuple
        with device_ctx():
            kernel = self.get_impl(which_impl)
            args = kernel.create_extargs(probe=True)
            d = torch.load(pt, map_location=default_device_string(), mmap=True)
            inputs = from_dict(data_class=kernel.PT_INPUT_CLASS, data=d["bidi_inputs"], config=dacite_tuple)
            _ = kernel(im, inputs, args)
            total_number_of_kernels = int(args.selected_kernel_total_hsacos)
            def gen():
                for hi in range(total_number_of_kernels):
                    args.set_hsaco(hsaco=hi, probe=True)
                    _ = kernel(im, inputs, args)
                    d = {
                        'psels': safeload(args.selected_hsaco_psels),
                        'copts': safeload(args.selected_hsaco_copts),
                    }
                    yield d
            return list(gen())

    def impl_desc(self, kernel, args) -> dict:
        from aotriton.tune.utils import safeload
        return {
            'psels': safeload(args.selected_hsaco_psels),
            'copts': safeload(args.selected_hsaco_copts),
        }
