# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/affine.<kernel_name>.{h,cc}

import io
from ..base import (
    typed_choice as TC,
    Functional,
)
from .interface import InterfaceGenerator
from ..affine import AffineDescription
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
