# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/op.<op_name>.{h,cc}

import io
from ..base import (
    typed_choice as TC,
    Functional,
    Interface,
)
from .interface import InterfaceGenerator
from ..op import Operator, MetroKernel, ConditionalKernel
from ..kernel import KernelDescription
from .template import get_template
from ..utils import (
    LazyFile,
)
from .common import codegen_struct_cfields, codegen_includes
from .optune import OptuneCodeGenerator

class OperatorGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('op.h')
    SOURCE_TEMPLATE = get_template('op.cc')
    KSHIM_LAUNCHER_TEMPLATE = get_template('kshim_launcher.cc')
    METRO_LAUNCHER_TEMPLATE = get_template('metro_launcher.cc')
    METRO_SNIPPET_TEMPLATE = get_template('snippet/metro_per_kernel.cc')
    IFELSE_SNIPPET_TEMPLATE = get_template('snippet/metro_per_kernel_ifelse.cc')
    PFX = 'iface'

    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame'):
        ocg = OptuneCodeGenerator(self._args, functional, df, self._this_repo)
        if not ocg.is_trivial:
            return ocg, True
        else:
            ocg.generate_trivial()
        return None, True

    def codegen_tune_struct_name(self, arch_number, godel_number):
        tt_dict = self._this_repo.get_data('trivial_tunes')
        trivial_enum = tt_dict.get((arch_number, godel_number), None)
        if trivial_enum is None:
            return super().codegen_tune_struct_name(arch_number, godel_number)
        tune_name = self._iface.TUNE_NAME
        is_extern = False
        return f'{tune_name}_{self._iface.NAME}__Trivial_{trivial_enum}', is_extern

    def write_shim_header(self, functionals, fout):
        iface = self._iface
        d = {
            'family_name'           : iface.FAMILY,
            'param_class_name'      : iface.param_class_name,
            'context_class_name'    : iface.context_class_name,
            'func_fields'           : codegen_struct_cfields(iface.func_cfields, nalign=4),
            'list_of_backend_enum'  : self.codegen_backend_enums(nalign=8),
            'fallback_backend'      : iface.fallback_backend.enum_name,
            'total_number_of_backends'      : self._iface.nbackends,
            'optune_table_entry_declares'   : self.codegen_tune_table_entry_declares(functionals),
            'number_of_functionals' : iface.godel_number,
            'declare_list_of_deduplicated_lut_functions' : '// TODO: declare_list_of_deduplicated_lut_functions' # self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        iface = self._iface
        d = {
            'family_name'               : iface.FAMILY,
            'iface_name'                : iface.NAME,
            'param_class_name'          : iface.param_class_name,
            'context_class_name'        : iface.context_class_name,
            'godel_number_body'         : self.codegen_godel_number_body(),
            'get_archmod_number_body'   : self.codegen_archmod_number_body(),
            'def_trivial_tunes'         : self.codegen_trivial_tunes(),
            'optune_table_entries'      : self.codegen_tune_table_entries(functionals),
            'number_of_functionals'     : iface.godel_number,
            'def_backend_launchers'     : self.codegen_launchers(nalign=0),
            'launcher_table_entries'    : self.codegen_launch_table_entries(nalign=4),
            'list_of_deduplicated_lut_functions' : '// TODO: list_of_deduplicated_lut_functions' # self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_backend_enums(self, nalign):
        stmt = []
        for i, backend in enumerate(self._iface.list_backends()):
            stmt.append(f'{backend.enum_name} = {i}')
        ALIGN = ',\n' + ' ' * nalign
        return ALIGN.join(stmt)

    def codegen_launchers(self, nalign):
        iface = self._iface
        stmt = []
        for backend in iface.list_backends():
            stmt.append(self.codegen_single_launcher(backend, nalign))
        ALIGN = '\n\n'
        return ALIGN.join(stmt)

    def codegen_single_launcher(self, backend : Interface, nalign):
        if isinstance(backend, KernelDescription):
            return self.codegen_kshim_launcher(backend, nalign)
        if isinstance(backend, MetroKernel):
            return self.codegen_metro_launcher(backend, nalign)
        assert False, f'Unsupported backend class {backend.__class__}'

    def codegen_kshim_launcher(self, kdesc : KernelDescription, nalign):
        iface = self._iface
        stmt = []
        self._add_header_for_source(kdesc)
        d = {
            'context_class_name'    : iface.context_class_name,
            'launcher_func_name'    : self.codegen_launcher_func_name(kdesc),
            'backend_context_name'  : kdesc.context_class_name,
        }
        return self.KSHIM_LAUNCHER_TEMPLATE.format_map(d)

    def codegen_metro_launcher(self, metro : MetroKernel, nalign):
        iface = self._iface
        context_class_name = iface.context_class_name
        stmt = []
        for kdesc in metro.list_kernels():
            if isinstance(kdesc, ConditionalKernel):
                self._add_header_for_source(kdesc.if_kernel)
                d = {
                    'condition'             : f'context.params->{kdesc.if_parameter} {kdesc.if_expr}',
                    'backend_context_name'  : kdesc.if_kernel.context_class_name,
                }
                if kdesc.else_kernel is None:
                    snippet = self.METRO_SNIPPET_TEMPLATE.format_map(d)
                else:
                    self._add_header_for_source(kdesc.else_kernel)
                    d['else_context_name'] = kdesc.else_kernel.context_class_name
                    snippet = self.IFELSE_SNIPPET_TEMPLATE.format_map(d)
            else:
                self._add_header_for_source(kdesc)
                d = {
                    'condition'             : 'true',
                    'backend_context_name'  : kdesc.context_class_name,
                }
                snippet = self.METRO_SNIPPET_TEMPLATE.format_map(d)
            stmt.append(snippet)
        stmt.append('return hipSuccess;')
        d = {
            'context_class_name'    : iface.context_class_name,
            'launcher_func_name'    : self.codegen_launcher_func_name(metro),
            'launch_every_kernel'   : '\n'.join(stmt),
        }
        return self.METRO_LAUNCHER_TEMPLATE.format_map(d)

    def codegen_launcher_func_name(self, backend):
        return f'launcher_for_{backend.enum_name}'

    def codegen_launch_table_entries(self, nalign):
        iface = self._iface
        stmt = [ '&' + self.codegen_launcher_func_name(b) for b in iface.list_backends() ]
        ALIGN = ',\n' + ' ' * nalign
        return ALIGN.join(stmt)

    def codegen_trivial_tunes(self):
        trivial_tunes = self._this_repo.get_data('trivial_tunes')
        uniques = set(trivial_tunes.values())
        context_class_name = self._iface.context_class_name
        tune_name = self._iface.TUNE_NAME
        stmt = []
        for trivial_enum in uniques:
            stmt.append(f'void {tune_name}_{self._iface.NAME}__Trivial_{trivial_enum}({context_class_name}& context, int) {{')
            stmt.append(f'    context.backend_index = {context_class_name}::BackendEnum::{trivial_enum};')
            stmt.append('}')
            stmt.append('')
        return '\n'.join(stmt)
