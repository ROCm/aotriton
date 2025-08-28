# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/shim.<kernel_name>.{h,cc}

import io
from ..base import (
    typed_choice as TC,
    Functional,
)
from .interface import InterfaceGenerator
from ..kernel import KernelDescription
from .template import get_template
from ..utils import (
    LazyFile,
    log
)
from .common import codegen_struct_cfields, codegen_includes
from .autotune import AutotuneCodeGenerator

class KernelShimGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('shim.h')
    SOURCE_TEMPLATE = get_template('shim.cc')
    PFX = 'shim'

    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame', sql : str):
        if functional.meta_object.is_functional_disabled(functional):
            log(lambda : f'Functional {functional.godel_number=} disabled')
            use_this_functional = False
            return None, use_this_functional
        use_this_functional = True
        return AutotuneCodeGenerator(self._args, functional, df, sql, self._this_repo), use_this_functional

    def write_shim_header(self, functionals, fout):
        kdesc = self._iface
        shared_iface = kdesc.SHARED_IFACE is not None
        if shared_iface:
            self._add_iface_for_source(kdesc.SHARED_IFACE)
            # hdr_name = kdesc.SHARED_IFACE.NAME
            # iface_header = f'#include "iface.{hdr_name}.h"'
        shared_iface_family = kdesc.SHARED_IFACE.FAMILY if shared_iface else kdesc.FAMILY
        d = {
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'param_class_name'      : kdesc.param_class_name,
            'shared_iface_family'   : shared_iface_family,
            'shared_iface'          : 1 if shared_iface else 0,
            'context_class_name'    : kdesc.context_class_name,
            'metadata_class_name'   : kdesc.metadata_class_name,
            'func_fields'           : codegen_struct_cfields(kdesc.func_cfields, nalign=4),
            'perf_fields'           : codegen_struct_cfields(kdesc.perf_cfields, nalign=4),
            'declare_compiled_in_features'  : self.codegen_declare_compiled_in_features(),
            'kernel_table_entry_declares'   : self.codegen_tune_table_entry_declares(functionals),
            'number_of_functionals' : kdesc._godel_number,
            'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._hdr_include_repo.get_data())
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        kdesc = self._iface
        shared_iface = kdesc.SHARED_IFACE is not None
        shared_iface_family = kdesc.SHARED_IFACE.FAMILY if shared_iface else kdesc.FAMILY
        list_of_pp_args_function_defs, list_of_pp_args_function_decls, pp_func_num = self.codegen_kernel_arguments()
        d = {
            'shared_iface'        : 1 if shared_iface else 0,
            'shared_iface_family' : shared_iface_family,
            'kernel_family_name'  : kdesc.FAMILY,
            'triton_kernel_name'  : kdesc.NAME,  # TODO: use signature so AMD_LOG_LEVEL=3 is more meaningful
            'shim_kernel_name'    : kdesc.NAME,
            'param_class_name'    : kdesc.param_class_name,
            'context_class_name'  : kdesc.context_class_name,
            'metadata_class_name' : kdesc.metadata_class_name,
            'godel_number_body'   : self.codegen_godel_number_body(),
            'pp_func_num'         : pp_func_num,
            'list_of_pp_args_function_defs' : list_of_pp_args_function_defs,
            'list_of_pp_args_function_decls' : list_of_pp_args_function_decls,
            'get_archmod_number_body' : self.codegen_archmod_number_body(),
            'number_of_functionals': kdesc._godel_number,
            'define_compiled_in_features' : self.codegen_define_compiled_in_features(),
            # 'copy_perf_fields_body': self.copy_perf_fields_body,
            'kernel_table_entry_declares' : self.codegen_tune_table_entry_declares(functionals),
            'per_kernel_packed_string'  : self.codegen_per_kernel_packed_string(),
            'kernel_table_entries' : self.codegen_tune_table_entries(functionals),
            'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
        }
        d['includes'] = codegen_includes(self._src_include_repo.get_data())
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_per_kernel_packed_string(self):
        return self._this_repo.get_data('per_kernel_packed_string')

    def codegen_declare_compiled_in_features(self):
        kdesc = self._iface
        decl_list = []
        for tp in kdesc.list_functional_params():
            tc = tp.repr_typed_choice
            if tp.maybe_conditional or tc.HIDDEN:
                continue
            infotype = tp.repr_typed_choice.infotype
            decl_code = f'static const std::vector<{infotype}>& get_{tp.repr_name}_choices();'
            decl_list.append(decl_code)
        return '\n    '.join(decl_list)

    def codegen_define_compiled_in_features(self):
        def_list = []
        kdesc = self._iface
        meta_class = kdesc.metadata_class_name
        for tp in kdesc.list_functional_params():
            tc = tp.repr_typed_choice
            if tp.maybe_conditional or tc.HIDDEN:
                continue
            infotype = tp.repr_typed_choice.infotype
            choices = ', '.join([tc.infotext for tc in tp.choices])
            def_code = f'''
const std::vector<{infotype}>& {meta_class}::get_{tp.repr_name}_choices()
{{
    static const std::vector<{infotype}> choices = {{ {choices} }};
    return choices;
}}'''
            def_list.append(def_code)
        return '\n'.join(def_list)

    def codegen_kernel_arguments(self):
        param_class_name = self._iface.param_class_name
        pp_registry = self._this_repo.get_data('pp_function')
        stmt = []
        # array = ['PP_FUNC prepare_arguments [] = {']
        array = []
        for assign_skips, (findex, src) in pp_registry.items():
            pp_function_name = f'{self._iface.NAME}_pp_args_{findex}'
            stmt.append(f'static std::vector<void*>')
            stmt.append(f'{pp_function_name}(const {param_class_name}& params,')
            stmt.append(' ' * len(pp_function_name) + ' const TritonAuxiliaryArguments& aux) {')
            stmt.append(src)
            stmt.append(f'}}')
            array.append(pp_function_name)
        pp_func_num = len(pp_registry.keys())
        return '\n'.join(stmt), ',\n  '.join(array), pp_func_num
