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
)
from .common import codegen_struct_cfields
from .autotune import AutotuneCodeGenerator

class KernelShimGenerator(InterfaceGenerator):
    HEADER_TEMPLATE = get_template('shim.h')
    SOURCE_TEMPLATE = get_template('shim.cc')
    PFX = 'shim'

    def create_sub_generator(self, functional : Functional, df : 'pandas.DataFrame'):
        return AutotuneCodeGenerator(self._args, functional, df, self._this_repo)

    def write_shim_header(self, functionals, fout):
        kdesc = self._iface
        not_shared = kdesc.SHARED_IFACE is None
        if not_shared:
            iface_header = '// No shared interface'
        else:
            hdr_name = kdesc.SHARED_IFACE.NAME
            iface_header = f'#include "iface.{hdr_name}.h"'
        d = {
            'kernel_family_name'    : kdesc.FAMILY,
            'shim_kernel_name'      : kdesc.NAME,
            'param_class_name'      : kdesc.param_class_name,
            'include_shared_iface'  : iface_header,
            'not_shared'            : 1 if not_shared else 0,
            'context_class_name'    : kdesc.context_class_name,
            'metadata_class_name'   : kdesc.metadata_class_name,
            'func_fields'           : codegen_struct_cfields(kdesc.func_cfields, nalign=4),
            'perf_fields'           : codegen_struct_cfields(kdesc.perf_cfields, nalign=4),
            'declare_compiled_in_features'  : self.codegen_declare_compiled_in_features(),
            'kernel_table_entry_declares'   : self.codegen_kernel_table_entry_declares(functionals),
            'number_of_functionals' : kdesc._godel_number,
            'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, functionals, fout):
        kdesc = self._iface
        list_of_pp_args_function_defs, list_of_pp_args_function_decls, pp_func_num = self.codegen_kernel_arguments()
        d = {
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
            'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(functionals),
            'per_kernel_packed_string'  : self.codegen_per_kernel_packed_string(),
            'kernel_table_entries' : self.codegen_kernel_table_entries(functionals),
            'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
        }
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


    def codegen_kernel_table_entry_declares(self, object_files):
        return '/* TODO: kernel_table_entry_declares */'

    def codegen_godel_number_body(self):
        body = io.StringIO()
        kdesc = self._iface
        for meta in kdesc.list_functional_params():
            self.codegen_godel_number_calculation(meta, body)
        return body.getvalue()

    def codegen_godel_number_calculation(self, meta, fout):
        if meta.nchoices <= 1:
            return
        aname = meta.repr_name # meta._ordered_arguments[0][0]
        INDENT = 4 * ' '
        print(INDENT + '{', file=fout)
        print(2 * INDENT + 'int64_t number = 0;', file=fout)
        for number, tc in enumerate(meta.choices):
            assert not isinstance(tc, TC.ConditionalChoice)
            if isinstance(tc, TC.tensor):
                type_enum = tc.type_enum
                print(2 * INDENT + f'if (args.{aname}->dtype() == {type_enum}) number = {number} ;', file=fout)
            else:
                value = str(tc).lower()
                print(2 * INDENT + f'if (args.{aname} == {value}) number = {number} ;', file=fout)
        print(2 * INDENT + f'sum += number * {meta.godel_number};', file=fout)
        print(1 * INDENT + '}', file=fout)

    def codegen_archmod_number_body(self):
        lets = []
        for i, arch in enumerate(self._target_arch_keys):
            for j, gpu in enumerate(self._target_arch[arch]):
                gpu_enum = f'GPU_AMD_ARCH_{gpu}'.upper()
                # CAVEAT: must return j because some GPU mod may not be selected.
                lets.append(f'if (gpu == {gpu_enum}) return {{ {i}, {j} }}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)

    def codegen_list_of_deduplicated_lut_functions(self):
        registry = self._this_repo.get_data('lut_function')
        stmt = [f'{fret} {fname} {fsrc};' for fsrc, (fret, fname, fparams) in registry.items()]
        return '\n'.join(stmt)

    def codegen_declare_list_of_deduplicated_lut_functions(self):
        registry = self._this_repo.get_data('lut_function')
        stmt = [f'extern {fret} {fname}{fparams};' for fsrc, (fret, fname, fparams) in registry.items()]
        return '\n'.join(stmt)

    def codegen_autotune_struct_name(self, arch_number, godel_number):
        return f'Autotune_{self._iface.NAME}__A{arch_number}__F{godel_number}'

    def codegen_kernel_table_entry_declares(self, functionals):
        decls = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            godel_numbers = sorted(list(set([f.godel_number for f in functionals])))
            for godel_number in godel_numbers:
                struct_name = self.codegen_autotune_struct_name(arch_number, godel_number)
                decls.append(f'void {struct_name}({self._iface.context_class_name}& params, int mod_number);')
        return '\n'.join(decls)

    def codegen_kernel_table_entries(self, functionals):
        lets = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            lets.append(4 * ' ' + '{')
            godel_numbers = sorted(list(set([f.godel_number for f in functionals])))
            for godel_number in range(self._iface.godel_number):
                struct_name = self.codegen_autotune_struct_name(arch_number, godel_number)
                if godel_number in godel_numbers:
                    lets.append(8 * ' ' + f'&autotune::{struct_name},')
                else:
                    lets.append(8 * ' ' + f'nullptr,')
            lets.append(4 * ' ' + '},')
        return '\n'.join(lets)

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
            stmt.append(' ' * len(pp_function_name) + ' hipDeviceptr_t* global_scratch) {')
            stmt.append(src)
            stmt.append(f'}}')
            array.append(pp_function_name)
        pp_func_num = len(pp_registry.keys())
        return '\n'.join(stmt), ',\n  '.join(array), pp_func_num
