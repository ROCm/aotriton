# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from ..base import (
    Interface,
    Functional,
)

class Operator(Interface):
    TUNE_NAME = 'optune'

    def __init__(self):
        super().__init__()
        self._late_init()

    @property
    def enum_name(self):
        # CamelName = self.NAME.replace('_', ' ').title().replace(' ', '')
        return f'kOp_{self.class_name_base}'

    def list_backends(self):
        return [] # TODO

    @property
    def nbackends(self):
        return 1 # TODO

    '''
    Operator is backed KernelDescription/MetroKernel/AffineKernel
    and has no additional params
    '''
    def list_non_functional_params(self):
        return []

    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        pass

    def translate_empty_dataframe(self, f : Functional):
        pass
