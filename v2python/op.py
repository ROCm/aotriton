# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Collection of Metro kernels is called operators

from .kernel_desc import (
    KernelDescription,
    get_template,
)
from .kernel_argument import (
    ArgumentCategory,
    ArgumentMetadata,
    ArgumentSelection,
)
from .gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus

class Operator(object):
    HEADER_TEMPLATE = get_template('op.h')
    SOURCE_TEMPLATE = get_template('op.cc')
    # Must define OP_NAME
    # OP_NAME

    def __init__(self, list_of_metro_kernels):
        assert self.OP_NAME is not None, f'class {self.__class__} misses OP_NAME'
        assert self.OP_ARGUMENTS, f'class {self.__class__} misses OP_ARGUMENTS'
        self._list_of_metro_kernels  = list_of_metro_kernels
        assert self._list_of_metro_kernels
        self._fallback_metro = None
        for k in self._list_of_metro_kernels:
            if k.is_fallback:
                self._fallback_metro = k
                break
        assert self._fallback_metro is not None
        self._set_of_kernel_shims = set(sum([k.individual_kernels for k in self._list_of_metro_kernels], []))
        self._target_arch = None
        self._target_arch_keys = None
        # We need a separate set of ArgumentMetadata because they may appear
        # with different orders in different kernels
        self._func_meta = []
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_TYPE, self) for k, v in self.TYPE_CHOICES.items()]
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_FEAT, self) for k, v in self.FEAT_CHOICES.items()]
        for m in self._func_meta:
            m.sort_arguments(self.OP_ARGUMENTS)
        self._func_meta = sorted(self._func_meta, key=lambda m: m.first_apperance)
        ArgumentMetadata.assign_godel_number(self._func_meta)
        self._godel_number = self._func_meta[0].godel_number * self._func_meta[0].nchoices
        self._func_selections = [m.spawn_all_selections() for m in self._func_meta]
        self._lut_lambda_registry = {}

    get_tensor_rank = KernelDescription.get_tensor_rank

    @property
    def target_arch(self):
        if self._target_arch is None:
            gpus = set.union(*[m.target_gpus for m in self._list_of_metro_kernels])
            self._target_arch = cluster_gpus(gpus)
        return self._target_arch

    @property
    def target_arch_keys(self):
        if self._target_arch_keys is None:
            self._target_arch_keys = list(self.target_arch.keys())
        return self._target_arch_keys

    @property
    def param_class_name(self):
        return "Op" + self.OP_NAME.replace('_', ' ').title().replace(' ', '') + 'Params'

    @property
    def context_class_name(self):
        return "Op" + self.OP_NAME.replace('_', ' ').title().replace(' ', '') + 'Context'

    @property
    def func_fields(self):
        return self._fallback_metro.func_fields

    def codegen_godel_number_body(self):
        body = io.StringIO()
        for m in self._func_meta:
            m.codegen_godel_number_calculation(body)
        return body.getvalue()

    def codegen_list_of_metro_kernels(self):
        ALIGN = '\n' + ' ' * 8
        metro_enums = [ f'{k.enum_name} = {i}' for i, k in enumerate(self._list_of_metro_kernels) ]
        return ALIGN.join(metro_enums)

    def codegen_list_of_shim_kernels(self):
        ALIGN = '\n' + ' ' * 8
        kernel_enums = [ f'{k.enum_name} = {i}' for i, k in enumerate(self._set_of_kernel_shims) ]
        return ALIGN.join(kernel_enums)

    def write_op_header(self, fout):
        d = { 'op_family_name'              : self.OP_FAMILY,
              'op_param_class_name'         : self.param_class_name,
              'op_context_class_name'       : self.context_class_name,
              'func_fields'                 : ';\n    '.join(self.func_fields),
              'list_of_metro_kernels'       : self.codegen_list_of_metro_kernels(),
              'total_number_of_metro_kernels'   : len(self._list_of_metro_kernels),
              'list_of_named_kernel_shim'       : self.codegen_list_of_shim_kernels(),
              'total_number_of_kernel_shims'    : len(self._set_of_kernel_shims),
              'number_of_functionals'       : self.godel_number,
              'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
              'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(),
            }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_op_source(self, fout):
        list_of_requirement_functions_defs, list_of_requirement_functions_decls = self.codegen_requirement_functions()
        d = { 'op_name'                     : self.OP_NAME,
              'include_shim_kernel_headers' : self.codegen_include(),
              'op_family_name'              : self.OP_FAMILY,
              'op_param_class_name'         : self.param_class_name,
              'op_context_class_name'       : self.context_class_name,
              'godel_number_body'           : self.codegen_godel_number_body(),
              'list_of_requirement_functions_defs' : list_of_requirement_functions_defs,
              'list_of_requirement_functions_decls' : list_of_requirement_functions_decls,
              'get_archmod_number_body'     : self.codegen_archmod_number_body(),
              'number_of_functionals'       : self._godel_number,
              'kernel_table_entries'        : self.codegen_kernel_table_entries(),
              'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
            }
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def codegen_requirement_functions(self):
        stmt = []
        array = []
        for mk in self._list_of_metro_kernels:
            def_stmt, fname = mk.codegen_requirement_function()
            if def_stmt:
                stmt.append(def_stmt)
                array.append('&{fname}')
            else:
                array.append('nullptr')
        return '\n'.join(stmt), ',\n  '.join(array)

    def codegen_archmod_number_body(self):
        for i, arch in enumerate(self.target_arch_keys):
            for j, gpu in enumerate(self.target_arch[arch]):
                gpu_enum = f'GPU_AMD_ARCH_{gpu}'.upper()
                # CAVEAT: must return j because some GPU mod may not be selected.
                lets.append(f'if (gpu == {gpu_enum}) return {{ {i}, {j} }}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)

    # TODO: Move LUT function into a separate, shared by all op kernels file.
    register_code_lut = KernelDescription.register_code_lut
    codegen_list_of_deduplicated_lut_functions = KernelDescription.codegen_list_of_deduplicated_lut_functions
    codegen_declare_list_of_deduplicated_lut_functions = KernelDescription.codegen_declare_list_of_deduplicated_lut_functions

    def codegen_kernel_table_entry_declares(self):
        decls = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            for godel_number in range(godel_numbers):
                struct_name = self.get_optune_struct_name(arch_number, godel_number)
                decls.append(f'void {struct_name}({self.param_class_name}& params, int mod_number);')
        return '\n'.join(decls)

    def codegen_include(self):
        headers = set(sum([k.codegen_dep_header_files for k in self._list_of_metro_kernels], []))
        return '\n'.join([f'#include "{h}"' for h in headers])

    def get_optune_struct_name(self, arch_number, godel_number):
        return f'Optune_{self.OP_NAME}__A{arch_number}__F{godel_number}'

    def codegen_kernel_table_entries(self):
        lets = []
        for arch_number, target_arch in enumerate(self.target_arch_keys):
            lets.append(4 * ' ' + '{')
            for godel_number in range(self._godel_number):
                # TODO: detect unsupported godel numbers, like Causal=True and Bias=True
                struct_name = self.get_optune_struct_name(arch_number, godel_number)
                if godel_number in godel_numbers:
                    lets.append(8 * ' ' + f'&optune::{struct_name},')
                else:
                    lets.append(8 * ' ' + f'nullptr,')
            lets.append(4 * ' ' + '},')
        return '\n'.join(lets)

    def gen_func_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._func_selections)

    def gen_tuned_op_lut(self, tuned_db : 'KernelTuningDatabase') -> 'Iterator[KernelTuningLutForGPU]':
        for (arch, gpus), fsels in itertools.product(self.target_arch.items(),
                                                     self.gen_func_selections()):
            dba = tuned_db.select_gpus(gpus)
            yield dba.arch, fsels, dba.get_lut(self, self.OPTUNE_KEYS, fsels, self._perf_meta)
