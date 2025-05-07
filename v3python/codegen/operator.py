# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/op.<op_name>.{h,cc}

import io
from .template import get_template
from ..utils import (
    LazyFile,
    RegistryRepository,
)
from .interface import InterfaceGenerator
from ..op import Operator
from ..gpu_targets import cluster_gpus

'''
TODO: Unify with KernelShimGenerator
'''
class OperatorGenerator(object):
    pass
# class OperatorGenerator(InterfaceGenerator):
#     HEADER_TEMPLATE = get_template('op.h')
#     SOURCE_TEMPLATE = get_template('op.cc')
#     PFX = 'op'
# 
#     def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame'):
#         return OptuneCodeGenerator(self._args, functional, df, self._this_repo)
# 
#     def write_shim_header(self, functionals, fout):
#         print(self.HEADER_TEMPLATE.format_map(d), file=fout)
# 
#     def write_shim_source(self, functionals, fout):
#         print(self.SOURCE_TEMPLATE.format_map(d), file=fout)
