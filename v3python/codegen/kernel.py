# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Generate <family>/shim.<kernel_name>.{h,cc}

import io
from ..base.conditional_value import ConditionalConstexpr
from ..op import NO_OPERATOR
from ..kernel.kdesc import KernelDescription
from ..utils import (
    LazyFile,
    get_template,
    RegistryRepository,
)
# from ..utils.is_tuning_enabled import is_tuning_on_for_kernel
from ..database import Factories as DatabaseFactories
from .common import codegen_struct_cfields
from .autotune import AutotuneCodeGenerator
from ..gpu_targets import cluster_gpus

class KernelShimGenerator(object):
    HEADER_TEMPLATE = get_template('shim.h')
    SOURCE_TEMPLATE = get_template('shim.cc')

    def __init__(self, args, k : KernelDescription):
        self._args = args
        self._kdesc = k
        # self._tuning = is_tuning_on_for_kernel(self._args, self._kdesc)
        self._target_gpus = args.target_gpus
        self._target_arch = cluster_gpus(self._target_gpus)
        self._target_arch_keys = list(self._target_arch.keys())
        print(f'{self._target_arch=}')

    def generate(self):
        # Un "self._" section
        args = self._args
        kdesc = self._kdesc

        # registry
        self._registry_repo = RegistryRepository()

        # autotune phase
        fac = DatabaseFactories.create_factory(args.build_dir)
        for functional in kdesc.gen_functionals(self._target_arch):
            df = fac.create_view(functional)
            AutotuneCodeGenerator(self._args, functional, df, self._registry_repo).generate()

        # shim code phase
        # Must be after autotune due to common functions needed by autotune is
        # generated in autotune module
        shim_path = args.build_dir / kdesc.FAMILY
        shim_path.mkdir(parents=True, exist_ok=True)
        shim_fn = 'shim.' + kdesc.NAME + '.h'
        fullfn = shim_path / shim_fn
        with LazyFile(fullfn) as fout:
            self.write_shim_header(fout)
        with LazyFile(fullfn.with_suffix('.cc')) as fout:
            self.write_shim_source(fout)

    def write_shim_header(self, fout):
        kdesc = self._kdesc
        empty_op = kdesc.OPERATOR == NO_OPERATOR
        d = {
            'kernel_family_name'  : kdesc.FAMILY,
            'shim_kernel_name'    : kdesc.NAME,
            'param_class_name'    : kdesc.param_class_name,
            'op_name'             : kdesc.op_name,
            'empty_op'            : 1 if empty_op else 0,
            'context_class_name'  : kdesc.context_class_name,
            'metadata_class_name' : kdesc.metadata_class_name,
            'func_fields'         : codegen_struct_cfields(kdesc.func_cfields, nalign=4),
            'perf_fields'         : codegen_struct_cfields(kdesc.perf_cfields, nalign=4),
            'declare_compiled_in_features' : self.codegen_declare_compiled_in_features(),
            'kernel_table_entry_declares' : 'TODO kernel_table_entry_declares', # self.codegen_kernel_table_entry_declares(object_files),
            'number_of_functionals': kdesc._godel_number,
            'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
        }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, fout):
        kdesc = self._kdesc
        op = kdesc.OPERATOR # TODO
        # list_of_pp_args_function_defs, list_of_pp_args_function_decls, pp_func_num = self.codegen_kernel_arguments()
        d = {
            'kernel_family_name'  : kdesc.FAMILY,
            'triton_kernel_name'  : kdesc.NAME,  # TODO: use signature so AMD_LOG_LEVEL=3 is more meaningful
            'shim_kernel_name'    : kdesc.NAME,
            'param_class_name'    : kdesc.param_class_name,
            'context_class_name'  : kdesc.context_class_name,
            'metadata_class_name' : kdesc.metadata_class_name,
            'godel_number_body'   : self.codegen_godel_number_body(),
            'pp_func_num'         : 'TODO pp_func_num', # pp_func_num,
            'list_of_pp_args_function_defs' : 'TODO list_of_pp_args_function_defs', # list_of_pp_args_function_defs,
            'list_of_pp_args_function_decls' : 'TODO list_of_pp_args_function_decls', # list_of_pp_args_function_decls,
            'get_archmod_number_body' : self.codegen_archmod_number_body(),
            'number_of_functionals': kdesc._godel_number,
            'define_compiled_in_features' : self.codegen_define_compiled_in_features(),
            # 'copy_perf_fields_body': self.copy_perf_fields_body,
            # 'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(object_files),
            'per_kernel_packed_string'  : self.codegen_per_kernel_packed_string(),
            'kernel_table_entries' : 'TODO kernel_table_entries', # self.codegen_kernel_table_entries(),
            'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
        }
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_per_kernel_packed_string(self):
        return self._registry_repo.get_data('per_kernel_packed_string')

    def codegen_declare_compiled_in_features(self):
        kdesc = self._kdesc
        decl_list = []
        for meta in kdesc.list_argument_metadata():
            if not meta.is_feature:
                continue
            ctype = meta.get_codegen_compiled_in_features_ctype()
            decl_code = f'static const std::vector<{ctype}>& get_{meta.repr_name}_choices();'
            decl_list.append(decl_code)
        return '\n    '.join(decl_list)

    def codegen_kernel_table_entry_declares(self, object_files):
        return '/* TODO: kernel_table_entry_declares */'

    def codegen_godel_number_body(self):
        body = io.StringIO()
        kdesc = self._kdesc
        for meta in kdesc.list_argument_metadata():
            self.codegen_godel_number_calculation(meta, body)
        return body.getvalue()

    def codegen_godel_number_calculation(self, meta, fout):
        if meta.nchoices <= 1:
            return
        triton_arg = meta.repr_name # meta._ordered_arguments[0][0]
        INDENT = 4 * ' '
        print(INDENT + '{', file=fout)
        print(2 * INDENT + 'int64_t number = 0;', file=fout)
        for number, choice in enumerate(meta.possible_choices):
            assert not isinstance(choice, ConditionalConstexpr)
            if meta.is_tensor:
                elem_type = choice[1:].split(':')[0]
                print(2 * INDENT + f'if (args.{triton_arg}->dtype() == {meta.DTYPE_NUMBER[elem_type]}) number = {number} ;', file=fout)
            else:
                value = str(choice).lower()
                print(2 * INDENT + f'if (args.{triton_arg} == {value}) number = {number} ;', file=fout)
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

    def codegen_define_compiled_in_features(self):
        def_list = []
        kdesc = self._kdesc
        meta_class = kdesc.metadata_class_name
        for meta in kdesc.list_argument_metadata():
            if not meta.is_feature:
                continue
            ctype = meta.get_codegen_compiled_in_features_ctype()
            choices = ', '.join(meta.get_codegen_compiled_in_features_values())
            def_code = f'''
const std::vector<{ctype}>& {meta_class}::get_{meta.repr_name}_choices()
{{
    static const std::vector<{ctype}> choices = {{ {choices} }};
    return choices;
}}'''
            def_list.append(def_code)
        return '\n'.join(def_list)

    def codegen_list_of_deduplicated_lut_functions(self):
        registry = self._registry_repo.get_data('lut_function')
        stmt = [f'{fret} {fname} {fsrc};' for fsrc, (fret, fname, fparams) in registry.items()]
        return '\n'.join(stmt)

    def codegen_declare_list_of_deduplicated_lut_functions(self):
        registry = self._registry_repo.get_data('lut_function')
        stmt = [f'extern {fret} {fname}{fparams};' for fsrc, (fret, fname, fparams) in registry.items()]
        return '\n'.join(stmt)
