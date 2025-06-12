# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from pathlib import Path
from ..gpu_targets import (
    cluster_gpus,
    gpu2arch,
    AOTRITON_ARCH_TO_PACK,
    AOTRITON_ARCH_TO_DIRECTORY,
    AOTRITON_TUNING_DATABASE_REUSE,
)
from ..utils import log

# Functional: describe the functional part of a certain compute process
# Abstract from: type choice + functional choice of Triton kernels
#                type choice + functional choice of operators
# Note:
# 1. Target GPU architecture is also part of a Functional
# 2. Arch number is assigned per-meta_object since it is possible some
#    kernel/operator is not supported on certain arch

def build_tc_dict(args):
    return { arg.name : arg.value for arg in args }

def build_compact_dict(args):
    return { arg.name : arg.value for arg in args if arg.show_in_compact }

def build_complete_dict(args):
    return { aname : arg for arg in args for aname in arg._klass.all_names }

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
        self._database_gpus = [ AOTRITON_TUNING_DATABASE_REUSE.get(gpu, gpu) for gpu in optimized_for ]
        self.__settle_conditional_values()
        self._compact_dict = build_compact_dict(self._binds)

    def __settle_conditional_values(self):
        while True:
            unresolved = [ bind for bind in self._binds if bind.is_unresolved ]
            if not unresolved:
                break
            tc_dict = build_tc_dict(self._binds)
            # tc_dict['__arch'] = self._arch
            log(lambda : f'{tc_dict=}')
            for bind in unresolved:
                bind.settle_unresolved(tc_dict)
                log(lambda : f'Settle {bind.name=} to {bind.value.triton_compile_signature=}')

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
    def database_gpus(self):
        return self._database_gpus

    @property
    def noptimized_for(self):
        return len(self._optimized_for)

    @property
    def meta_object(self):
        return self._meta

    @property
    def godel_number(self):
        return sum([s.godel_number for s in self._binds])

    '''
    dict of repr -> typed choice
    Smaller one for resolution
    '''
    def build_tc_dict(self):
        return build_tc_dict(self._binds)

    '''
    dict of all parameter -> typed choice
    '''
    def build_complete_bind_dict(self, with_resolved_tc=False):
        d = build_complete_dict(self._binds)
        if not with_resolved_tc:
            return d
        return { aname : (bind, bind.get_typed_value(aname)) for aname, bind in d.items() }

    def build_complete_tc_dict(self, with_resolved_tc=False):
        d = build_complete_dict(self._binds)
        return { aname : bind.get_typed_value(aname) for aname, bind in d.items() }

    @property
    def human_readable_signature(self):
        lf = [s.human_readable_signature for s in self._binds]
        return 'Human-readable Signature \n// ' + '\n// '.join([x for x in lf if x is not None])

    @property
    def compact_choices(self) -> dict:
        return self._compact_dict

    @property
    def fallback_choices(self) -> dict:
        return self.meta_object.fallback_compact_dict(self._compact_dict)

    '''
    "core" signature
    only directly used to supply TritonKernel as HSACO name component
    '''
    @property
    def signature_in_func_name(self):
        lf = [bind.signature_in_func_name for bind in self._binds if bind.show_in_compact]
        return '_'.join([x for x in lf])

    '''
    file pack signature only cares about Functional, so it is FONLY__
    '''
    @property
    def filepack_signature(self):
        sf = self.signature_in_func_name
        pack = AOTRITON_ARCH_TO_PACK.get(self.arch, self.arch)
        return 'FONLY__' + sf + f'___{pack}'

    '''
    Unlike filepack which may consolates multiple arches into the same file.
    Each arch has its own tunecc file.
    '''
    @property
    def tunecc_signature(self):
        sf = self.signature_in_func_name
        return 'FONLY__' + sf + f'___{self.arch}'

    '''
    Used by KSignature to construct full kernel name
    '''
    @property
    def compact_signature_noarch(self):
        sf = self.signature_in_func_name
        return 'F__' + sf

    @property
    def full_filepack_dir(self):
        return Path(AOTRITON_ARCH_TO_DIRECTORY[self.arch]) / self._meta.FAMILY / self._meta.NAME

    @property
    def full_filepack_path(self):
        return self.full_filepack_dir / self.filepack_signature

    def translate_dataframe(self, df):
        # Only meta object (kdesc/op) of a functional knows how to translate dataframe
        return self.meta_object.translate_dataframe(self, df)
