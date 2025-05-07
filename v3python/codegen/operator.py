# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/op.<op_name>.{h,cc}

import io
from ..base import (
    typed_choice as TC,
    Functional,
)
from .interface import InterfaceGenerator
from ..op import Operator
from .template import get_template
from ..utils import (
    LazyFile,
)
from .common import codegen_struct_cfields

'''
TODO: Unify with KernelShimGenerator
'''
class OperatorGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('op.h')
    SOURCE_TEMPLATE = get_template('op.cc')
    PFX = 'iface'

    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame'):
        return None
        # return OptuneCodeGenerator(self._args, functional, df, self._this_repo)

    def write_shim_header(self, functionals, fout):
        iface = self._iface
        d = {
            'op_family_name'        : iface.FAMILY,
            'param_class_name'      : iface.param_class_name,
            'context_class_name'    : iface.context_class_name,
            'func_fields'           : codegen_struct_cfields(iface.func_cfields, nalign=4),
            'list_of_backend_enum'  : self.codegen_backend_enums(nalign=8),
            'total_number_of_backends'      : self._iface.nbackends,
            'kernel_table_entry_declares'   : self.codegen_kernel_table_entry_declares(functionals),
            'number_of_functionals' : iface.godel_number,
            'declare_list_of_deduplicated_lut_functions' : 'TODO' # self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        return # TODO
        d = {
            'op_family_name'        : iface.FAMILY,
            'param_class_name'      : iface.param_class_name,
            'context_class_name'    : iface.context_class_name,
            'func_fields'           : codegen_struct_cfields(iface.func_cfields, nalign=4),
            'kernel_table_entry_declares'   : self.codegen_kernel_table_entry_declares(functionals),
            'number_of_functionals' : iface.godel_number,
            'declare_list_of_deduplicated_lut_functions' : 'TODO' # self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_backend_enums(self, nalign):
        stmt = []
        for i, backend in enumerate(self._iface.list_backends()):
            stmt.append(f'{backend.enum_name} = {i}')
        ALIGN = ',\n' + ' ' * nalign
        return ALIGN.join(stmt)
