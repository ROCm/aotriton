# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from collections import defaultdict
import io
import os
from pathlib import Path
from .conditional_value import (
    ConditionalConstexpr,
    ConditionalDeferredConstexpr,
    ConditionalDeferredElseTensor,
)
from .kernel_argument import (
    ArgumentCategory,
    ArgumentMetadata,
    ArgumentSelection,
)
from .kernel_signature import KernelSignature
from .object_desc import ObjectFileDescription
from .gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus

SOURCE_PATH = Path(__file__).resolve()
AOTRITON_ENABLE_FP32 = bool(int(os.getenv('AOTRITON_ENABLE_FP32', True)))

# We use [[ ]] instead of { } for C++ code template
def get_template(name):
    with open(SOURCE_PATH.parent.parent / 'v2src' / 'template' / name, 'r') as f:
        return f.read().replace('{', '{{').replace('}', '}}').replace('[[', '{').replace(']]', '}')

def join_dicts(dicts : 'list[dict]') -> dict:
    return { k:v for d in dicts for k,v in d.items() }

def get_possible_choices(klass, arg_name : str) -> 'list[Any]':
    for d in [klass.TYPE_CHOICES, klass.FEAT_CHOICES, klass.PERF_CHOICES]:
        for k, v in d.items():
            if arg_name in k:
                return v
    assert False, f"cannot find {arg_name}"

def select_pattern(arguments, prefix, trim_left=None, trim_right=None, delete_when=None):
    ret = []
    for s in arguments:
        assert s.strip() == s, f'Input argument {s} within {arguments=} contains spaces at either end'
        if s.startswith(prefix):
            ret.append(s)
    return (ret[trim_left:trim_right], delete_when)

class KernelDescription(object):
    ARGUMENTS = []
    SHIM_KERNEL_NAME = None
    _ARGUMENT_CHOICES = None
    HEADER_TEMPLATE = get_template('shim.h')
    SOURCE_TEMPLATE = get_template('shim.cc')
    MAIN_DATATYPES = ['*fp16:16', '*bf16:16', '*fp32:16'] if AOTRITON_ENABLE_FP32 else ['*fp16:16', '*bf16:16']

    TYPE_CHOICES = {
    }
    FEAT_CHOICES = {
    }
    PERF_CHOICES = {
    }

    @property
    def FULL_KERNEL_NAME(self):
        return f'{self.KERNEL_FAMILY}.{self.SHIM_KERNEL_NAME}'

    @property
    def ARGUMENT_CHOICES(self):
        if self._ARGUMENT_CHOICES is None:
            self._ARGUMENT_CHOICES = join_dicts([self.TYPE_CHOICES, self.FEAT_CHOICES, self.PERF_CHOICES])
        return self._ARGUMENT_CHOICES

    @property
    def KERNEL_DATA_ARGUMENTS(self):
        if self._DATA_ARGUMENTS is None:
            def is_data_argument(a):
                for k in self.TYPE_CHOICES.keys():
                    if a in k:
                        return True
                return False
            self._DATA_ARGUMENTS = [ a for a in self.ARGUMENTS if is_data_argument(a) ]
            print(f'{self._DATA_ARGUMENTS=}')
            if False:  # Don't, their names are translated inside codegen_kernel_arguments
                # Patch tensor
                for m in self._func_meta:
                    if not m.is_tensor:
                        continue
                    for a in m._ordered_arguments:
                        i = self._DATA_ARGUMENTS.index(a[0])
                        self._DATA_ARGUMENTS[i] += '_ptr'
        return self._DATA_ARGUMENTS

    def is_functional_disabled_on_arch(self, arch, fsels):
        return False

    def insert_tensor_strides_to_choices(self, last_is_continuous=False):
        for tensor, (strides, delete_when) in self.TENSOR_STRIDE_INPUTS.items():
            typed_strides = strides[:-1] if last_is_continuous else strides
            if delete_when is None:
                stride_dtype = 'u64:8'
            else:
                feat, feat_value = delete_when
                stride_dtype = ConditionalConstexpr(feat, feat_value, 0, 'u64:8')
            self.TYPE_CHOICES[frozenset(typed_strides)] = [stride_dtype]
            constant_strides = [] if not last_is_continuous else strides[-1:]
            if constant_strides:
                self.FEAT_CHOICES[frozenset(constant_strides)] = [1]
        print(f"{self.TYPE_CHOICES=}")
        print(f"{self.FEAT_CHOICES=}")

    def __init__(self, triton_kernel_name, triton_file_path):
        self.insert_tensor_strides_to_choices(last_is_continuous=True)
        self._DATA_ARGUMENTS = None
        self._triton_file_path = Path(triton_file_path)
        self._triton_kernel_name = triton_kernel_name
        self._func_meta = []
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_TYPE, self) for k, v in self.TYPE_CHOICES.items()]
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_FEAT, self) for k, v in self.FEAT_CHOICES.items()]
        self._perf_meta = [ArgumentMetadata(k, v, ArgumentCategory.CAT_PERF, self) for k, v in self.PERF_CHOICES.items()]
        for m in self._func_meta:
            m.sort_arguments(self.ARGUMENTS)
            for u, fallback in self.PARTIALLY_TUNED_FUNCTIONALS:
                if m.has_argument(u):
                    m.set_incomplete_tuning(fallback)
        self._func_meta = sorted(self._func_meta, key=lambda m: m.first_apperance)
        # print(f'{self._func_meta}')
        ArgumentMetadata.assign_godel_number(self._func_meta)
        # The godel number for architectures
        self._godel_number = self._func_meta[0].godel_number * self._func_meta[0].nchoices
        for m in self._perf_meta:
            m.sort_arguments(self.ARGUMENTS)
        # self._perf_meta = sorted(self._perf_meta, key=lambda m: m.first_apperance)
        # Note: Re-order with byte sizes can reduce the C-struct size in autotune files.
        self._perf_meta = sorted(self._perf_meta, key=lambda m : m.param_cc_size, reverse=True)
        # Performance arguments do not need godel numbers, they will be handled in a different way
        # ArgumentMetadata.assign_godel_number(self._perf_meta)
        self._func_selections = [m.spawn_all_selections() for m in self._func_meta]
        self._perf_selections = [m.spawn_all_selections() for m in self._perf_meta]
        self._target_gpus = None
        self._target_arch = None
        self.AUTOTUNE_KEYS_VALIDATED = []
        for key in self.ARGUMENTS:
            if key not in self.AUTOTUNE_KEYS:
                continue
            is_type = False
            for type_keys in self.TYPE_CHOICES.keys():
                if key in type_keys:
                    is_type = True
                    break
            if is_type:
                self.AUTOTUNE_KEYS_VALIDATED.append((key, self.AUTOTUNE_KEYS[key]))
        '''
        AUTOTUNE_KEYS sanity check, otherwise autotune code may be broken (already happened twice).
        '''
        for key in self.AUTOTUNE_KEYS:
            assert key in self.ARGUMENTS, f'AUTOTUNE_KEYS "{key}" cannot be found in {self.__class__.__name__}.ARGUMENTS'
        self._lut_lambda_registry = {}
        self._prepare_args_registry = {}
        self._kargs_getter_dic = None

    def set_target_gpus(self, gpus):
        # Note _target_gpus should not bu used
        # self._target_gpus = list(gpus)
        self._target_arch = cluster_gpus(gpus)
        self._target_arch_keys = list(self._target_arch.keys())

    @property
    def name(self):
        return self._triton_kernel_name

    def gen_func_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._func_selections)

    def gen_perf_selections(self, gpu, fsels) -> 'tuple[ArgumentSelection]':
        # Function options is handled at KernelSignature
        return itertools.product(*self._perf_selections)

    # DispatcherV3 Note:
    #   Only called by gen_all_object_files to select perf options for all matched GPUs
    def __gen_tuned_perf_selections(self,
                                    tuned_db : 'KernelTuningDatabase',
                                    arch : str,
                                    gpus : list[str],
                                    fsels : 'list[ArgumentSelection]'):
        dba = tuned_db.select_gpus(gpus)

        if dba.empty:  # Fallback to selection defined by KernelDescription
            for gpu in gpus:
                for psels in self.gen_perf_selections(gpu, fsels):
                    yield gpu, fsels, psels, None
                    break  # For empty tuning database. Only need one option
            return

        for gpu, psels, compiler_options in dba.select(fsels, self._perf_meta):
            yield gpu, fsels, psels, compiler_options

    def _gen_all_options_from_kdesc_autotune_configs(self,
                                                     gpu : str,
                                                     fsels : 'list[ArgumentSelection]'):
        fsel_dict = ArgumentSelection.build_fsel_dict(fsels)
        for cfg in self.gen_autotune_configs(gpu, fsel_dict):
            psels, compiler_options = cfg.translate_to_psel_and_co(self._perf_meta)
            yield gpu, fsels, psels, compiler_options

    def gen_all_object_files(self,
                             outpath : Path,
                             # kernel_name : str = None,
                             # file_name_prefix : str = None,
                             tuned_db : 'KernelTuningDatabase',
                             sancheck_fileexists = False) -> 'Iterator[ObjectFileDescription]':
        # DispatcherV3 Note:
        #   This function generate all hsaco objects, for tuning or for running
        #   Therefore, arch is always used.
        assert tuned_db is not None, '[KernelDescription.gen_all_object_files] Must pass not None tuned_db'
        def gen():
            if tuned_db.build_for_tuning:
                if hasattr(self, 'gen_autotune_configs'):
                    # Build for Tuning, for complex kernels wait for tuning
                    for arch, fsels in itertools.product(self._target_arch.keys(),
                                                         self.gen_func_selections()):
                        yield from self._gen_all_options_from_kdesc_autotune_configs(arch, fsels)
                else:
                    # FIXME: This yield is incorrect (missing gpu and fsels)
                    #        but apparently not triggering anything wrong right now.
                    for arch, fsels in itertools.product(self._target_arch.keys(),
                                                        self.gen_func_selections()):
                        yield from itertools.product(self.gen_perf_selections(arch, fsels),
                                                     [None])
                return
            # Not Build for Tuning, checking database
            for arch, gpus in self._target_arch.items():
                for fsels in self.gen_func_selections():
                    if self.is_functional_disabled_on_arch(arch, fsels):
                        # Empty tuning database
                        # Disabling the compiling is done in generate_compile.py
                        for psels in self.gen_perf_selections(arch, fsels):
                            yield arch, fsels, psels, None
                            break
                    else:
                        yield from self.__gen_tuned_perf_selections(tuned_db, arch, gpus, fsels)
        debug_counter = 0
        for gpu, fsels, psels, compiler_options in gen():
            try:
                sig = KernelSignature(self, fsels, psels, compiler_options, gpu)
            except Exception as e:
                print(f"{fsels=}")
                print(f"{psels=}")
                import traceback
                traceback.print_exc()
                exit()
            yield self.build_object_file_description(outpath, sig, sancheck_fileexists=sancheck_fileexists)
            if False: # Debugging
                debug_counter += 1
                if debug_counter > 10:
                    break

    def build_object_file_description(self, outpath, sig, sancheck_fileexists=False):
        return ObjectFileDescription(self, sig, outpath, sancheck_fileexists=sancheck_fileexists)

    # DispatcherV3 Note
    # LUT object now handles different GPU mods under the same arch
    def gen_tuned_kernel_lut(self, tuned_db : 'KernelTuningDatabase') -> 'Iterator[KernelTuningLutForGPU]':
        for (arch, gpus), fsels in  itertools.product(self._target_arch.items(),
                                                      self.gen_func_selections()):
            dba = tuned_db.select_gpus(gpus)
            # print(f'gen_tuned_kernel_lut {fsels=}')
            yield dba.arch, fsels, dba.get_lut(self, self.AUTOTUNE_KEYS_VALIDATED, fsels, self._perf_meta)

    @property
    def param_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_")) + 'Params'

    @property
    def context_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_")) + 'Context'

    @property
    def metadata_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_")) + 'Metadata'

    @property
    def func_fields(self):
        unsorted_fields = sum([m.param_cc_fields_tuple for m in self._func_meta], [])
        print(f'{self.SHIM_KERNEL_NAME} {unsorted_fields=}')
        fields = sorted(unsorted_fields, key=lambda tup: tup[2])
        return [cc_type + ' ' + aname for (cc_type, aname, _) in fields]

    @property
    def perf_fields(self):
        return sum([m.param_cc_fields for m in self._perf_meta], [])

    def write_shim_header(self, fout, object_files):
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'shim_kernel_name'    : self.SHIM_KERNEL_NAME,
              'param_class_name'    : self.param_class_name,
              'context_class_name'  : self.context_class_name,
              'metadata_class_name' : self.metadata_class_name,
              'func_fields'         : ';\n    '.join(self.func_fields),
              'perf_fields'         : ';\n    '.join(self.perf_fields),
              'declare_compiled_in_features' : self.codegen_declare_compiled_in_features(),
              'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(object_files),
              'number_of_functionals': self._godel_number,
              'declare_list_of_deduplicated_lut_functions' : self.codegen_declare_list_of_deduplicated_lut_functions(),
            }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_shim_source(self, fout, object_files, noimage_mode):
        list_of_pp_args_function_defs, list_of_pp_args_function_decls, pp_func_num = self.codegen_kernel_arguments()
        if not noimage_mode:
            assert self.SHIM_KERNEL_NAME == object_files[0].binary_entrance
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'triton_kernel_name'  : self.SHIM_KERNEL_NAME,
              'shim_kernel_name'    : self.SHIM_KERNEL_NAME,
              'param_class_name'    : self.param_class_name,
              'context_class_name'  : self.context_class_name,
              'metadata_class_name' : self.metadata_class_name,
              'godel_number_body'   : self.godel_number_body,
              'pp_func_num'         : pp_func_num,
              'list_of_pp_args_function_defs' : list_of_pp_args_function_defs,
              'list_of_pp_args_function_decls' : list_of_pp_args_function_decls,
              'get_archmod_number_body' : self.codegen_archmod_number_body(),
              'number_of_functionals': self._godel_number,
              'define_compiled_in_features' : self.codegen_define_compiled_in_features(),
              # 'copy_perf_fields_body': self.copy_perf_fields_body,
              # 'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(object_files),
              'kernel_table_entries' : self.codegen_kernel_table_entries(object_files),
              'list_of_deduplicated_lut_functions' : self.codegen_list_of_deduplicated_lut_functions(),
            }
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def get_tensor_rank(self, tensor_arg):
        if self.SHIM_KERNEL_NAME == 'bwd_kernel_dq':
            ret = self.TENSOR_RANKS.get(tensor_arg, self.TENSOR_RANKS['_default'])
            print(f"{tensor_arg=} {ret}")
        return self.TENSOR_RANKS.get(tensor_arg, self.TENSOR_RANKS['_default'])

    '''
    Create a superset dict
    codegen_deduplicated_pp_args_function_index will use a subset
    '''
    def codegen_kargs_getter_dic(self):
        if self._kargs_getter_dic is not None:
            return self._kargs_getter_dic
        d = {}
        for tensor_aname, (stride_anames, _) in self.TENSOR_STRIDE_INPUTS.items():
            tensor_rank = self.get_tensor_rank(tensor_aname)
            for i in range(tensor_rank):
                aname = stride_anames[i]
                d[aname] = f'params.{tensor_aname}->kparam_stride({i})'
        for m in self._func_meta:
            if not m.is_tensor:
                continue
            for aname in m.argument_names:
                d[aname] = f'params.{aname}->kparam_data_ptr()'
        for aname in self.KERNEL_DATA_ARGUMENTS:
            if aname in d:
                continue
            d[aname] = f'CAST(&params.{aname})'
        self._kargs_getter_dic = d
        return self._kargs_getter_dic

    def codegen_kernel_arguments(self):
        param_class_name = self.param_class_name
        stmt = []
        # array = ['PP_FUNC prepare_arguments [] = {']
        array = []
        for assign_skips, (pp_index, src, pp_function_name) in self._prepare_args_registry.items():
            stmt.append(f'static std::vector<void*>')
            stmt.append(f'{pp_function_name}(const {param_class_name}& params,')
            stmt.append(' ' * len(pp_function_name) + ' hipDeviceptr_t* global_scratch) {')
            stmt.append(src)
            stmt.append(f'}}')
            array.append(pp_function_name)
        pp_func_num = len(self._prepare_args_registry.keys())
        return '\n'.join(stmt), ',\n  '.join(array), pp_func_num

    @property
    def godel_number_body(self):
        body = io.StringIO()
        for m in self._func_meta:
            m.codegen_godel_number_calculation(body)
        return body.getvalue()

    def get_arch_number(self, arch : str) -> int:
        return self._target_arch_keys.index(arch)

    def codegen_archmod_number_body(self):
        lets = []
        for i, arch in enumerate(self._target_arch_keys):
            for j, gpu in enumerate(self._target_arch[arch]):
                gpu_enum = f'GPU_AMD_ARCH_{gpu}'.upper()
                # CAVEAT: must return j because some GPU mod may not be selected.
                lets.append(f'if (gpu == {gpu_enum}) return {{ {i}, {j} }}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)

    '''
    @property
    def copy_perf_fields_body(self):
        lets = []
        for field in self.perf_fields:
            lets.append(f'param.{field} = {field}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)
    '''

    def incbin_mangle(self, arch, o):
        return f'INCBIN_{arch}_{self.KERNEL_FAMILY}_{self.SHIM_KERNEL_NAME}_{o.c_identifier_signature}'

    def codegen_kernel_table_entries_per_arch(self, arch, object_files):
        ALIGN0 = '\n' + 4 * ' '
        ALIGN1 = '\n' + 8 * ' '
        ALIGN2 = '\n' + 12 * ' '
        d = defaultdict(list)
        lets = []
        for o in object_files:
            godel_number = o.godel_number
            image_symbol = self.incbin_mangle(arch, o)
            initializer_list = [f'.kernel_image = {image_symbol}']
            initializer_list += o.designated_perf_initializer_list
            d[godel_number].append('{ ' + ', '.join(initializer_list) + ' }')
            # d[godel_number].append(self.get_single_kernel_table_entry(arch, o))
        for k, v in d.items():
            lets.append(f'[{k}] = {{' +
                        ALIGN2 +
                        (','+ALIGN2).join(v) +
                        ALIGN1 + '},')
        return '{ ' + ALIGN1 + ALIGN1.join(lets) + ALIGN0 + '}'

    def get_single_kernel_table_entry(self, arch : 'str', o : 'ObjectFileDescription'):
        image_symbol = self.incbin_mangle(arch, o)

    def get_autotune_struct_name(self, arch_number, godel_number):
        return f'Autotune_{self.SHIM_KERNEL_NAME}__A{arch_number}__F{godel_number}'

    # def get_autotune_superclass_name(self):
    #     return f'Autotune_{self.SHIM_KERNEL_NAME}__Superclass'

    def codegen_kernel_table_entry_declares(self, object_files):
        decls = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            godel_numbers = sorted(list(set([o.godel_number for o in object_files if o.target_arch == target_arch])))
            for godel_number in godel_numbers:
                struct_name = self.get_autotune_struct_name(arch_number, godel_number)
                decls.append(f'void {struct_name}({self.param_class_name}& params, int mod_number);')
        return '\n'.join(decls)

    def codegen_kernel_table_entries(self, object_files):
        lets = []
        for arch_number, target_arch in enumerate(self._target_arch_keys):
            lets.append(4 * ' ' + '{')
            godel_numbers = sorted(list(set([o.godel_number for o in object_files])))
            for godel_number in range(self._godel_number):
                struct_name = self.get_autotune_struct_name(arch_number, godel_number)
                if godel_number in godel_numbers:
                    lets.append(8 * ' ' + f'&autotune::{struct_name},')
                else:
                    lets.append(8 * ' ' + f'nullptr,')
            lets.append(4 * ' ' + '},')
        return '\n'.join(lets)

    def sancheck_lut_tensor(self,
                            gpu,
                            lut_tensor,
                            fsels : 'list[ArgumentSelection]'):
        raise NotImplemented(f'{self.__class__}.sancheck_lut_tensor')

    def codegen_declare_compiled_in_features(self):
        decl_list = []
        for meta in self._func_meta:
            if not meta.is_feature:
                continue
            ctype = meta.get_codegen_compiled_in_features_ctype()
            decl_code = f'static const std::vector<{ctype}>& get_{meta.repr_name}_choices();'
            decl_list.append(decl_code)
        return '\n    '.join(decl_list)

    def codegen_define_compiled_in_features(self):
        def_list = []
        meta_class = self.metadata_class_name
        for meta in self._func_meta:
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

    def register_code_lut(self, lambda_src : str, lut_dtype : str, lut_shape : str):
        if lambda_src in self._lut_lambda_registry:
            return self._lut_lambda_registry[lambda_src][0]
        findex = len(self._lut_lambda_registry)
        lut_function_name = f'{self.SHIM_KERNEL_NAME}__lut_lambda_{findex}'
        self._lut_lambda_registry[lambda_src] = (lut_function_name, lut_dtype, lut_shape)
        return lut_function_name

    def lookup_prepare_args(self, assign_skips):
        if assign_skips in self._prepare_args_registry:
            return True, self._prepare_args_registry[assign_skips][0]
        return False, -1

    def register_prepare_args(self, assign_skips, pp_src : str):
        if assign_skips in self._prepare_args_registry:
            return self._prepare_args_registry[assign_skips][0]
        findex = len(self._prepare_args_registry)
        pp_function_name = f'{self.SHIM_KERNEL_NAME}_pp_args_{findex}'
        self._prepare_args_registry[assign_skips] = (findex, pp_src, pp_function_name)
        return findex

    def codegen_list_of_deduplicated_lut_functions(self):
        param_class_name = self.param_class_name
        stmt = []
        for src, (lut_function_name, lut_dtype, lut_shape) in self._lut_lambda_registry.items():
            stmt.append(f'int {lut_function_name} {src}\n')
        return '\n'.join(stmt)

    def codegen_declare_list_of_deduplicated_lut_functions(self):
        param_class_name = self.param_class_name
        stmt = []
        for _, (lut_function_name, lut_dtype, lut_shape) in self._lut_lambda_registry.items():
            stmt.append(f'extern int {lut_function_name}({param_class_name}&, int, {lut_dtype} {lut_shape});')
        return '\n'.join(stmt)
