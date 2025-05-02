# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from ..gpu_targets import cluster_gpus, gpu2arch
from .argument import build_dict, build_compact_dict, build_complete_dict

# Functional: describe the functional part of a certain compute process
# Abstract from: type choice + functional choice of Triton kernels
#                type choice + functional choice of operators
# Note:
# 1. Target GPU architecture is also part of a Functional
# 2. Arch number is assigned per-meta_object since it is possible some
#    kernel/operator is not supported on certain arch


class Functional(object):

    def __init__(self,
                 meta_object,  # KernelDescription | Operator
                 arch,
                 arch_number,
                 binds,
                 optimized_for):
        self._arch = arch
        self._arch_number = arch_number
        self._meta = meta_object
        self._binds = binds
        self._optimized_for = optimized_for
        self.__settle_conditional_values()
        self._fsel_dict = build_dict(self._binds)
        self._compact_dict = build_compact_dict(self._binds)

    def __settle_conditional_values(self):
        while True:
            unresolved = [ bind for bind in self._binds if bind.is_unresolved ]
            if not unresolved:
                break
            bind_dict = build_dict(self._binds)
            # bind_dict['__arch'] = self._arch
            for bind in unresolved:
                bind.settle_unresolved(bind_dict)

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
    def noptimized_for(self):
        return len(self._optimized_for)

    @property
    def meta_object(self):
        return self._meta

    @property
    def godel_number(self):
        return sum([s.godel_number for s in self._binds])

    @property
    def fsel_dict(self):
        return build_complete_dict(self._binds)

    '''
    dict of all parameter -> argument
    '''
    @property
    def complete_dict(self):
        return build_complete_dict(self._binds)

    @property
    def human_readable_signature(self):
        lf = [s.human_readable_signature for s in self._binds]
        return 'Human-readable Signature \n// ' + '\n// '.join([x for x in lf if x is not None])

    @property
    def compact_choices(self) -> dict:
        return self._compact_dict

    '''
    "core" signature
    only directly used to supply TritonKernel as HSACO name component
    '''
    @property
    def signature_in_func_name(self):
        lf = [s.compact_signature for s in self._binds if s.show_in_compact]
        return '_'.join([x for x in lf])

    '''
    file pack signature only cares about Functional, so it is FONLY__
    '''
    @property
    def filepack_signature(self):
        sf = self.signature_in_func_name
        return 'FONLY__' + sf + f'___{self.arch}'

    '''
    Used by KSignature to construct full kernel name
    '''
    @property
    def compact_signature(self):
        sf = self.signature_in_func_name
        return 'F__' + sf + f'___{self.arch}'

    @property
    def full_filepack_path(self):
        return Path(self._meta.FAMILY) / self._meta.NAME / self.filepack_signature

    def translate_dataframe(self, df):
        # Only meta object (kdesc/op) of a functional knows how to translate dataframe
        return self.meta_object.translate_dataframe(self, df)
