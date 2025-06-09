# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/affine.<kernel_name>.{h,cc}

import io
from ..base import (
    typed_choice as TC,
    Functional,
)
from .interface import InterfaceGenerator
from ..affine import AffineKernelDescription
from ..utils import (
    LazyFile,
    log
)

class AffineGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('affine.h')
    SOURCE_TEMPLATE = get_template('affine.cc')
    PFX = 'affine'

    def __init__(self, args, iface : Interface, parent_repo : RegistryRepository):
        super().__init__(args, iface, parent_repo)
        adesc = iface
        # Patch _target_arch since affine kernel may not support all arches.
        self._target_arch = { arch: gpus in self._target_arch.items() if arch in adesc.SUPPORTED_ARCH }
        del self._target_gpus  # For safety

    def create_sub_generator(self, functional : Functional):
        akdesc = functional.meta_object
        if akdesc.is_functional_disabled(functional):
            log(lambda : f'Functional {functional.godel_number=} disabled in affine kernel {akdesc.NAME}')
            use_this_functional = False
            return None, use_this_functional
        use_this_functional = True
        return AffineAsmGenerator(self._args, functional, self._this_repo), use_this_functional

    def write_shim_header(self, functionals, fout):
        pass

    def write_shim_source(self, functionals, fout):
        pass

    def codegen_godel_number_body(self):
        body = io.StringIO()
        iface = self._iface
        for tp in iface.list_functional_params():
            self.codegen_godel_number_calculation(tp, body)
        for tp in iface.list_residual_functional_params():
            self.codegen_godel_number_calculation(tp, body)
        return body.getvalue()
        return body.getvalue()
