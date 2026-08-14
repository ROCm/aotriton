# Copyright © 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
attn_options wrapper and KernelControl mixin for flash attention kernel selection.

Contains AttnOptionsWrapper (manages KernelControl bits for HSACO selection)
and SdpaOpts (mixin providing create_extargs / KERNEL_SLOT for tuning kernels).
Only safe to import from the tuning library (installed/<arch>/lib).
"""

from pyaotriton.v3 import KernelControl
from pyaotriton.v3.flash import attn_options


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
        hsaco_index = which_impl.hsaco_index if which_impl is not None else None
        ext = self.EXT_CLASS(self.BACKEND_INDEX, self.KERNEL_SLOT)
        ext.set_hsaco(hsaco=hsaco_index, probe=probe)
        return ext

    @property
    def KERNEL_SLOT(self):
        return int(getattr(self.EXT_CLASS.C_CLASS, self.__class__.__name__))
