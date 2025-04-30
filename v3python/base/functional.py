# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from ..gpu_targets import cluster_gpus, gpu2arch
from .parameter import build_dict, build_compact_dict, build_complete_dict

# Functional: describe the functional part of a certain compute process
# Abstract from: type choice + functional choice of Triton kernels
#                type choice + functional choice of operators
# Note:
# 1. Target GPU architecture is also part of a Functional
# 2. Arch number is assigned per-meta_object since it is possible some
#    kernel/operator is not supported on certain arch


class Functional(object):

    def __init__(self,
                 arch,
                 arch_number,
                 meta_object,  # KernelDescription | Operator
                 selections,
                 optimized_for):
        self._arch = arch
        self._arch_number = arch_number
        self._meta = meta_object
        self._selections = selections
        self._optimized_for = optimized_for
        self.__settle_conditional_values()
        self._fsel_dict = build_dict(self._selections)
        self._compact_dict = build_compact_dict(self._selections)

    def __settle_conditional_values(self):
        while True:
            unresolved = [ sel for sel in self._selections if sel.is_unresolved ]
            if not unresolved:
                break
            sel_dict = build_dict(self._selections)
            for sel in unresolved:
                sel.settle_unresolved(self._arch, sel_dict)

    @property
    def arch(self):
        return self._arch

    @property
    def arch_number(self):
        return self._arch_number

    @property
    def optimized_for(self):
        return self._optimized_for

    @property
    def meta_object(self):
        return self._meta

    @property
    def fsel_dict(self):
        return build_complete_dict(self._selections)

    @property
    def complete_dict(self):
        return build_complete_dict(self._selections)

    @property
    def human_readable_signature(self):
        lf = [s.human_readable_signature for s in self._selections]
        return '#if 0 // Human-readable Signature \n' + '\n '.join([x for x in lf if x is not None]) + '\n#endif'

    @property
    def compact_choices(self) -> dict:
        return self._compact_dict

    '''
    Note here we use FONLY__
    file pack signature only cares about Functional
    '''
    @property
    def filepack_signature(self):
        lf = [s.compact_signature for s in self._selections]
        sf = '_'.join([x for x in lf if x is not None])
        return 'FONLY__' + sf + f'___{self.arch}'

    @property
    def compact_signature(self):
        lf = [s.compact_signature for s in self._selections]
        sf = '_'.join([x for x in lf if x is not None])
        return 'F__' + sf + f'___{self.arch}'

    @property
    def full_filepack_path(self):
        return Path(self._meta.FAMILY) / self._meta.NAME / self.filepack_signature

    def translate_dataframe(self, df):
        # Only meta object (kdesc/op) of a functional knows how to translate dataframe
        return self.meta_object.translate_dataframe(self, df)
