import itertools
from collections import defaultdict
import io
from pathlib import Path
from .kernel_argument import (
    ArgumentCategory,
    ArgumentMetadata,
    ArgumentSelection
)
from .kernel_signature import KernelSignature
from .object_desc import ObjectFileDescription
from .gpu_targets import AOTRITON_SUPPORTED_GPUS

SOURCE_PATH = Path(__file__).resolve()

# We use [[ ]] instead of { } for C++ code template
def get_template(name):
    with open(SOURCE_PATH.parent.parent / 'v2src' / 'template' / name, 'r') as f:
        return f.read().replace('{', '{{').replace('}', '}}').replace('[[', '{').replace(']]', '}')

def join_dicts(dicts : 'list[dict]') -> dict:
    return { k:v for d in dicts for k,v in d.items() }

def get_possible_types(klass, arg_name : str) -> 'list[str]':
    for d in [klass.TYPE_CHOICES, klass.FEAT_CHOICES, klass.PERF_CHOICES]:
        for k, v in d.items():
            if arg_name in k:
                return v
    assert False, f"cannot find {arg_name}"

def select_pattern(arguments, prefix, trim_left=None, trim_right=None):
    ret = []
    for s in arguments:
        if s.startswith(prefix):
            ret.append(s)
    return ret[trim_left:trim_right]

class KernelDescription(object):
    ARGUMENTS = []
    SHIM_KERNEL_NAME = None
    _ARGUMENT_CHOICES = None
    HEADER_TEMPLATE = get_template('launcher.h')
    SOURCE_TEMPLATE = get_template('launcher.cc')

    TYPE_CHOICES = {
    }
    FEAT_CHOICES = {
    }
    PERF_CHOICES = {
    }

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

    def insert_tensor_strides_to_choices(self, last_is_continuous=False):
        for tensor, strides in self.TENSOR_STRIDE_INPUTS.items():
            typed_strides = strides[:-1] if last_is_continuous else strides
            self.TYPE_CHOICES[frozenset(typed_strides)] = ['u64:16']
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
        self._perf_meta = sorted(self._perf_meta, key=lambda m: m.first_apperance)
        # Performance arguments do not need godel numbers, they will be handled in a different way
        # ArgumentMetadata.assign_godel_number(self._perf_meta)
        self._func_selections = [m.spawn_all_selections() for m in self._func_meta]
        self._perf_selections = [m.spawn_all_selections() for m in self._perf_meta]
        self._target_gpus = None
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

    def gen_func_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._func_selections)

    def gen_perf_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._perf_selections)

    def gen_tuned_perf_selections(self,
                                  tuned_db : 'KernelTuningDatabase',
                                  gpu : str,
                                  fsels : 'list[ArgumentSelection]'):
        dba = tuned_db.select_gpu(gpu, self._target_gpus.index(gpu))
        for psels, compiler_options in dba.select(fsels, self._perf_meta):
            yield gpu, fsels, psels, compiler_options

    def set_target_gpus(self, gpus):
        self._target_gpus = ['native'] if gpus is None else list(gpus)

    def gen_all_object_files(self,
                             outpath : Path,
                             # kernel_name : str = None,
                             # file_name_prefix : str = None,
                             tuned_db : 'KernelTuningDatabase' = None,
                             sancheck_fileexists = False) -> 'Iterator[ObjectFileDescription]':
        def gen():
            if tuned_db is None or tuned_db.empty:
                yield from itertools.product(self._target_gpus,
                                             self.gen_func_selections(),
                                             self.gen_perf_selections(),
                                             [None])
            else:
                for gpu, fsels in itertools.product(self._target_gpus,
                                                    self.gen_func_selections()):
                    yield from self.gen_tuned_perf_selections(tuned_db, gpu, fsels)
        debug_counter = 0
        for gpu, fsels, psels, compiler_options in gen():
            sig = KernelSignature(self, fsels, psels, compiler_options, gpu)
            yield self.build_object_file_description(outpath, sig, sancheck_fileexists=sancheck_fileexists)
            if False: # Debugging
                debug_counter += 1
                if debug_counter > 10:
                    break

    def build_object_file_description(self, outpath, sig, sancheck_fileexists=False):
        # print(f"{gpu=} {fsels=} {psels=} {compiler_options=}")
        # sig = KernelSignature(self, fsels, psels, compiler_options, gpu)
        # fn = file_name_prefix + '-Kernel-' if file_name_prefix else ''
        # kernel_name =  if kernel_name is None else kernel_name
        fn = self.SHIM_KERNEL_NAME
        # print(f'{sig.compact_signature=}')
        fn += '-Sig-' + sig.compact_signature
        fn += '-Gpu-' + sig.target_gpu
        fn += '.hsaco'
        return ObjectFileDescription(self, sig, outpath / fn, sancheck_fileexists=sancheck_fileexists)

    def gen_tuned_kernel_lut(self, tuned_db : 'KernelTuningDatabase') -> 'Iterator[KernelTuningLutForGPU]':
        for gpu, fsels in itertools.product(self._target_gpus,
                                            self.gen_func_selections()):
            dba = tuned_db.select_gpu(gpu, self._target_gpus.index(gpu))
            # print(f'gen_tuned_kernel_lut {fsels=}')
            yield gpu, fsels, dba.get_lut(self, self.AUTOTUNE_KEYS_VALIDATED, fsels, self._perf_meta)

    @property
    def param_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_")) + 'Params'

    @property
    def context_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_")) + 'Context'

    @property
    def func_fields(self):
        return sum([m.param_cc_fields for m in self._func_meta], [])

    @property
    def perf_fields(self):
        return sum([m.param_cc_fields for m in self._perf_meta], [])

    def write_launcher_header(self, fout, object_files):
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'shim_kernel_name'    : self.SHIM_KERNEL_NAME,
              'param_class_name'    : self.param_class_name,
              'context_class_name'  : self.context_class_name,
              'func_fields'         : ';\n    '.join(self.func_fields),
              'perf_fields'         : ';\n    '.join(self.perf_fields),
              'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(object_files),
              'number_of_functionals': self._godel_number,
            }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_launcher_source(self, fout, object_files):
        put_kernel_arguments_on_stack, let_kernel_arguments = self.codegen_kernel_arguments()
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'triton_kernel_name'  : object_files[0].binary_entrance,
              'shim_kernel_name'    : self.SHIM_KERNEL_NAME,
              'param_class_name'    : self.param_class_name,
              'context_class_name'  : self.context_class_name,
              'godel_number_body'   : self.godel_number_body,
              'put_kernel_arguments_on_stack' : put_kernel_arguments_on_stack,
              'let_kernel_arguments' : let_kernel_arguments,
              'get_arch_number_body' : self.arch_number_body,
              'number_of_functionals': self._godel_number,
              # 'copy_perf_fields_body': self.copy_perf_fields_body,
              # 'kernel_table_entry_declares' : self.codegen_kernel_table_entry_declares(object_files),
              'kernel_table_entries' : self.codegen_kernel_table_entries(object_files),
            }
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def get_tensor_rank(self, tensor_arg):
        return self.TENSOR_RANKS_OVERRIDE.get(tensor_arg, self.TENSOR_RANKS_OVERRIDE['_default'])

    def codegen_kernel_arguments(self):
        stack_lets = []
        stack_variables = {}
        for tensor_aname, stride_anames in self.TENSOR_STRIDE_INPUTS.items():
            tensor_rank = self.get_tensor_rank(tensor_aname)
            for i in range(tensor_rank - 1):
                aname = stride_anames[i]
                stack_lets.append(f'uint64_t {aname} = params.{tensor_aname}->stride({i})')
                stack_variables[aname] = aname
        for m in self._func_meta:
            if not m.is_tensor:
                continue
            for aname in m.argument_names:
                stack_lets.append(f'const void* {aname}_ptr = params.{aname}->data_ptr()')
                stack_variables[aname] = f'{aname}_ptr';
        ALIGN = ',\n' + ' ' * 32
        def plet(aname):
            if aname in stack_variables.keys():
                sname = stack_variables[aname]
            else:
                sname = f'params.{aname}'
            return f'const_cast<void*>(static_cast<const void*>(&{sname}))'
        lets = [plet(aname) for aname in self.KERNEL_DATA_ARGUMENTS]
        return ';\n    '.join(stack_lets), ALIGN.join(lets)

    @property
    def godel_number_body(self):
        body = io.StringIO()
        for m in self._func_meta:
            m.codegen_godel_number_calculation(body)
        return body.getvalue()

    @property
    def arch_number_body(self):
        lets = []
        for i, gpu in enumerate(self._target_gpus):
            arch = AOTRITON_SUPPORTED_GPUS[gpu]
            lets.append(f'if (arch == {arch}) return {i}')
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

    def codegen_kernel_table_entry_declares(self, object_files):
        decls = []
        for arch_number, target_gpu in enumerate(self._target_gpus):
            godel_numbers = sorted(list(set([o.godel_number for o in object_files if o.target_gpu == target_gpu])))
            for godel_number in godel_numbers:
                struct_name = self.get_autotune_struct_name(arch_number, godel_number)
                decls.append(f'struct {struct_name} {{')
                decls.append(f'    void operator()({self.param_class_name}& params);')
                decls.append(f'}};')
        return '\n'.join(decls)

    def codegen_kernel_table_entries(self, object_files):
        lets = []
        for arch_number, target_gpu in enumerate(self._target_gpus):
            lets.append(4 * ' ' + '{')
            godel_numbers = sorted(list(set([o.godel_number for o in object_files])))
            for godel_number in range(self._godel_number):
                struct_name = self.get_autotune_struct_name(arch_number, godel_number)
                if godel_number in godel_numbers:
                    lets.append(8 * ' ' + f'autotune::{struct_name}(),')
                else:
                    lets.append(8 * ' ' + f'[]({self.param_class_name}&) {{}},')
            lets.append(4 * ' ' + '},')
        return '\n'.join(lets)

