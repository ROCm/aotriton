# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import itertools
from collections import defaultdict
import io
import os
from pathlib import Path
from ..base.functional import (
    Functional,
)
from ..base.conditional_value import (
    ConditionalConstexpr,
    ConditionalDeferredConstexpr,
    ConditionalDeferredElseTensor,
)
from ..base.argument import (
    TypeParameter as TParam,
    ValueParameter as VParam,
    Argument as Arg,
)
from ..base.ttype import (
    typename_t,
    guess_vparam_type,
)
from ..op import (
    Operator,
    NO_OPERATOR,
)
from .ksignature import KernelSignature
from .object_desc import ObjectFileDescription
from ..gpu_targets import AOTRITON_SUPPORTED_GPUS, cluster_gpus
from ..utils import get_template

SOURCE_PATH = Path(__file__).resolve()
AOTRITON_ENABLE_FP32 = bool(int(os.getenv('AOTRITON_ENABLE_FP32', True)))

def join_dicts(dicts : 'list[dict]') -> dict:
    return { k:v for d in dicts for k,v in d.items() }

def get_possible_choices(klass, arg_name : str) -> 'list[Any]':
    l = []
    for k in ['TYPE_CHOICES', 'FEAT_CHOICES', 'PERF_CHOICES']:
        if hasattr(klass, k):
            l += [getattr(klass, k)]
    for d in l:
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

def collect_functionals_from_op(klass):
    mklass = klass.OPERATOR
    if mklass is None and klass.CONFIRM_NO_OPERATOR:
        return
    print(f'collect_functionals_from_op {klass=} {mklass=}')
    assert mklass, f'Class {klass} must define OPERATOR'
    # Early detection
    all_assigned = True
    for which_choice in ['TYPE_CHOICES', 'FEAT_CHOICES']:
        if getattr(klass, which_choice, None) is None:
            all_assigned = False
            break
    if all_assigned:
        return
    args_order = { aname : i for i, aname in enumerate(klass.ARGUMENTS) }
    args_in_use = set(klass.ARGUMENTS)
    print(f'{args_order=}')
    # Selection is defined in Op but not all options are available in individual kernels
    def remove_missing(args_to_determine : frozenset):
        print(f'intersection {args_to_determine=} vs {args_in_use=}')
        args_to_determine = set(args_to_determine).intersection(args_in_use)
        print(f'result {args_to_determine=}')
        sorted_args = sorted(args_to_determine, key = lambda aname : args_order[aname])
        print(f'result {sorted_args=}')
        return tuple(sorted_args)  # TODO: replace frozenset with tuple

    CHOICE_FILTERS = klass.CHOICE_FILTERS
    # remove_unsupported(('Q', 'K', 'V'), [16, 32, 64]) = [16], when CHOICE_FILTERS = { 'K' : lambda x : x < 32 }
    def remove_unsupported(key, values):
        if not CHOICE_FILTERS:
            return values
        for k in key:
            if k in CHOICE_FILTERS:
                return [ v for v in values if CHOICE_FILTERS[k](v) ]
        return values

    for which_choice in ['TYPE_CHOICES', 'FEAT_CHOICES']:
        if getattr(klass, which_choice, None) is not None:
            continue
        mattr = getattr(mklass, which_choice)
        dic = {}
        for k, v in mattr.items():
            args = remove_missing(k)
            v = remove_unsupported(k, v)
            if args:
                dic[args] = v
        setattr(klass, which_choice, dic)
        print(f"{klass}'s final {which_choice} is {dic}")

    for which_choice in ['TENSOR_RANKS', 'TENSOR_STRIDE_INPUTS']:
        if getattr(klass, which_choice, None) is not None:
            continue
        mattr = getattr(mklass, which_choice)
        dic = {}
        for k, v in mattr.items():
            if k in args_in_use:
                dic[k] = v
        setattr(klass, which_choice, dic)
    klass.TENSOR_RANKS['_default'] = mklass.TENSOR_RANKS['_default']

class KernelDescription(object):
    COMPILER_OPTIONS = [ 'waves_per_eu', 'num_warps', 'num_stages' ]
    OPERATOR = None
    ARGUMENTS = []
    NAME = None
    _ARGUMENT_CHOICES = None
    HEADER_TEMPLATE = get_template('shim.h')
    SOURCE_TEMPLATE = get_template('shim.cc')

    # Type and Feature are shared from Related Op
    # TYPE_CHOICES = {
    # }
    # FEAT_CHOICES = {
    # }
    PERF_CHOICES = {
    }
    # Exclude unsupported combinations
    CHOICE_FILTERS = {
    }

    @property
    def FULL_KERNEL_NAME(self):
        return f'{self.FAMILY}.{self.NAME}'

    # FIXME: Unify with param_class_name
    @property
    def enum_name(self):
        CamelName = self.NAME.replace('_', ' ').title().replace(' ', '')
        return f'kShim_{CamelName}'

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
        collect_functionals_from_op(self.__class__)
        self.insert_tensor_strides_to_choices(last_is_continuous=True)
        self._DATA_ARGUMENTS = None
        self._triton_file_path = Path(triton_file_path)
        self._triton_kernel_name = triton_kernel_name
        self._func_meta = []
        def __ttype(aname):
            return ttype.tensor_type(self.get_tensor_rank(aname))
        self._func_meta += [ TParam(k, [guess_tparam_type(t) for t in v], ttype=ttype.typename_t) for k, v in self.TYPE_CHOICES.items()]
        self._func_meta += [VParam(k, v, ttype=guess_vparam_type(v)) for k, v in self.FEAT_CHOICES.items()]
        self._perf_meta = [VParam(k, v) for k, v in self.PERF_CHOICES.items()]
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

    def list_argument_metadata(self):
        yield from self._func_meta

    def gen_functionals(self, target_arch):
        for arch_number, arch in enumerate(target_arch.keys()):
            gpus = target_arch[arch]
            for fsels in itertools.product(*self._func_selections):
                yield Functional(self, arch, arch_number, fsels, optimized_for=gpus)

    # TODO: dataframe name mangling should be deferred to database package.
    #       Possible solution is to attach a translator to DataFrame object
    def translate_dataframe(self, f : Functional, df : 'pandas.DataFrame'):
        sparse_keys = [ f'inputs${key}' for key in self.AUTOTUNE_KEYS_VALIDATED ]
        binning_dict = { key : algo(df[f'inputs${key}'].unique()) for key, algo in self.AUTOTUNE_KEYS_VALIDATED.item() }
        sparse_key_possible_values = { key : sorted(df[key].unique()) for key in sparse_keys }
        # sparse_shape is not used because lut is compact
        lut_shape = [ len(sparse_key_possible_values[key]) for key in sparse_keys ]
        # lut starts with a large enough dtype
        lut_tensor = np.empty(lut_shape, dtype=np.int32)
        perf_keys = [ 'tuned_kernel${meta.repr_name}' for meta in self._perf_meta ]
        copt_keys = [ 'compiler_options${key}' for key in self.COMPILER_OPTIONS ]
        def perf_bind(series):
            return [ meta.create_direct(value) for meta, value in zip(self._perf_meta, series[perf_keys]) ]
        sigs = []
        sigs_dict = {}
        def discretization(tup):
            key, value = tup
            return sparse_key_possible_values[key].index(value)
        def find_ind(series):
            sparse = series[sparse_keys]
            return [ discretization(key, value) for key, value in zip(sparse_keys, sparse) ]
        def register_sig(series):
            key = tuple(list(series[perf_keys]) + list(series[copt_keys]))
            if key not in sigs_dict:
                sigs_dict[key] = len(sigs)
                sig = KernelSignature(f,
                                      perf_bind(series)[perf_keys],
                                      series[copt_keys])
                sigs.append(sig)
            return sigs_dict[sig.values_tuple]
        for idx, series in df.iterrows():
            lut_ind = find_ind(series)
            sig_num = register_sig(series)
            lut_tensor[lut_ind] = sig_num
        nsigs = len(sigs)
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            if nsigs < np.iinfo(dtype).max:
                break
        lut_tensor = lut_tensor.astype(dtype)
        # self.sancheck_lut_tensor(lut_tensor)
        return lut_tensor, sigs, binning_dict

class __OLD__KERNEL__DESCRIPTION(object):

    def set_target_gpus(self, gpus):
        # Note _target_gpus should not bu used
        # self._target_gpus = list(gpus)
        self._target_gpus = gpus
        self._target_arch = cluster_gpus(gpus)
        self._target_arch_keys = list(self._target_arch.keys())

    @property
    def name(self):
        return self._triton_kernel_name

    @property
    def target_gpus(self):
        return self._target_gpus

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
        for (arch, gpus), fsels in itertools.product(self._target_arch.items(),
                                                     self.gen_func_selections()):
            dba = tuned_db.select_gpus(gpus)
            # print(f'gen_tuned_kernel_lut {fsels=}')
            yield dba.arch, fsels, dba.get_lut(self, self.AUTOTUNE_KEYS_VALIDATED, fsels, self._perf_meta)

    @property
    def param_class_name(self):
        return "".join(x.capitalize() for x in self.NAME.lower().split("_")) + 'Params'

    @property
    def op_name(self):
        if self.OPERATOR == NO_OPERATOR:
            return None
        return self.OPERATOR.NAME

    @property
    def param_class_name(self):
        if self.OPERATOR == NO_OPERATOR:
            return self.class_name_base + 'Params'
        return self.OPERATOR.param_class_name

    @property
    def class_name_base(self):
        return "".join(x.capitalize() for x in self.NAME.lower().split("_"))

    @property
    def context_class_name(self):
        return self.class_name_base + 'Context'

    @property
    def metadata_class_name(self):
        return self.class_name_base + 'Metadata'

    '''
    _cfields means the data type has been translated to c types
    '''
    @property
    def func_cfields(self):
        unsorted_fields = sum([m.param_cc_fields_tuple for m in self._func_meta], [])
        print(f'{self.NAME} {unsorted_fields=}')
        fields = sorted(unsorted_fields, key=lambda tup: tup[2])
        return [ (cc_type, aname) for (cc_type, aname, _) in fields]

    @property
    def perf_cfields(self):
        fields = sum([m.param_cc_fields_tuple for m in self._perf_meta], [])
        return [ (cc_type, aname) for (cc_type, aname, _) in fields]

    def get_tensor_rank(self, tensor_arg):
        if self.NAME == 'bwd_kernel_dq':
            ret = self.TENSOR_RANKS.get(tensor_arg, self.TENSOR_RANKS['_default'])
            print(f"{tensor_arg=} {ret}")
        print(f'{self=} {self.TENSOR_RANKS=}')
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
        return f'INCBIN_{arch}_{self.KERNEL_FAMILY}_{self.NAME}_{o.c_identifier_signature}'

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
        return f'Autotune_{self.NAME}__A{arch_number}__F{godel_number}'

    # def get_autotune_superclass_name(self):
    #     return f'Autotune_{self.NAME}__Superclass'

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
        lut_function_name = f'{self.NAME}__lut_lambda_{findex}'
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
        pp_function_name = f'{self.NAME}_pp_args_{findex}'
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
