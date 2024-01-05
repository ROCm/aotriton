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

def select_pattern(arguments, prefix):
    ret = []
    for s in arguments:
        if s.startswith(prefix):
            ret.append(s)
    return ret

class KernelDescription(object):
    ARGUMENTS = []
    SHIM_KERNEL_NAME = None
    _ARGUMENT_CHOICES = None
    _DATA_ARGUMENTS = None
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
            # Patch tensor
            for m in self._func_meta:
                if not m.is_tensor:
                    continue
                for a in m._ordered_arguments:
                    i = self._DATA_ARGUMENTS.index(a[0])
                    self._DATA_ARGUMENTS[i] += '_ptr'
        return self._DATA_ARGUMENTS

    def __init__(self, triton_kernel_name, triton_file_path):
        self._triton_file_path = Path(triton_file_path)
        self._triton_kernel_name = triton_kernel_name
        self._func_meta = []
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_TYPE, self) for k, v in self.TYPE_CHOICES.items()]
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_FEAT, self) for k, v in self.FEAT_CHOICES.items()]
        self._perf_meta = [ArgumentMetadata(k, v, ArgumentCategory.CAT_PERF, self) for k, v in self.PERF_CHOICES.items()]
        for m in self._func_meta:
            m.sort_arguments(self.ARGUMENTS)
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

    def gen_func_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._func_selections)

    def gen_perf_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._perf_selections)

    def gen_tuned_perf_selections(self,
                                  tuned_db : 'KernelTuningDatabase',
                                  gpu : str,
                                  fsels : 'list[ArgumentSelection]'):
        dba = tuned_db.select_gpu(gpu)
        for psels, compiler_options in dba.select(fsels, self._perf_meta):
            yield gpu, fsels, psels, compiler_options

    def set_target_gpus(self, gpus):
        self._target_gpus = ['native'] if gpus is None else list(gpus)

    def gen_all_object_files(self,
                             outpath : Path,
                             kernel_name : str = None,
                             file_name_prefix : str = None,
                             tuned_db : 'KernelTuningDatabase' = None) -> 'list[ObjectFileDescription]':
        kernel_name = self.SHIM_KERNEL_NAME if kernel_name is None else kernel_name
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
            # print(f"{gpu=} {fsels=} {psels=} {compiler_options=}")
            sig = KernelSignature(self, fsels, psels, compiler_options, gpu)
            fn = file_name_prefix + '-Kernel-' if file_name_prefix else ''
            fn += kernel_name
            # print(f'{sig.compact_signature=}')
            fn += '-Sig-' + sig.compact_signature
            fn += '-Gpu-' + gpu
            fn += '.hsaco'
            yield ObjectFileDescription(self, sig, outpath / fn)
            if False: # Debugging
                debug_counter += 1
                if debug_counter > 10:
                    break

    @property
    def param_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_")) + 'Params'

    @property
    def func_fields(self):
        return sum([m.param_cc_fields for m in self._func_meta], [])

    @property
    def perf_fields(self):
        return sum([m.param_cc_fields for m in self._perf_meta], [])

    def write_launcher_header(self, fout):
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'param_class_name'    : self.param_class_name,
              'func_fields'         : ';\n    '.join(self.func_fields),
              'perf_fields'         : ';\n    '.join(self.perf_fields),
              'number_of_functionals': self._godel_number,
            }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_launcher_source(self, fout, object_files):
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'shim_kernel_name'    : self.SHIM_KERNEL_NAME,
              'param_class_name'    : self.param_class_name,
              'godel_number_body'   : self.godel_number_body,
              'let_tensor_stride_arguments' : self.let_tensor_stride_arguments,
              'let_kernel_arguments' : self.let_kernel_arguments,
              'get_arch_number_body' : self.arch_number_body,
              'number_of_functionals': self._godel_number,
              'copy_perf_fields_body': self.copy_perf_fields_body,
              'kernel_table_entries' : self.get_kernel_table_entries(object_files),
            }
        print(self.SOURCE_TEMPLATE.format_map(d), file=fout)

    def get_tensor_rank(self, tensor_arg):
        return self.TENSOR_RANKS_OVERRIDE.get(tensor_arg, self.TENSOR_RANKS_OVERRIDE['_default'])

    @property
    def let_tensor_stride_arguments(self):
        lets = []
        for k, v in self.TENSOR_STRIDE_INPUTS.items():
            tensor_rank = self.get_tensor_rank(k)
            for i in range(tensor_rank):
                lets.append(f'uint64_t {v[i]} = {k}.stride({i})')
        for m in self._func_meta:
            if not m.is_tensor:
                continue
            for a in m._ordered_arguments:
                lets.append(f'const void* {a[0]}_ptr = {a[0]}.data_ptr()')
        return ';\n    '.join(lets)

    @property
    def let_kernel_arguments(self):
        ALIGN = ',\n' + ' ' * 32
        lets = [f'static_cast<void*>(&{a})' for a in self.KERNEL_DATA_ARGUMENTS]
        return ALIGN.join(lets)

    @property
    def godel_number_body(self):
        body = io.StringIO()
        for m in self._func_meta:
            m.write_godel_number_calculation(body)
        return body.getvalue()

    @property
    def arch_number_body(self):
        lets = []
        for i, gpu in enumerate(self._target_gpus):
            arch = AOTRITON_SUPPORTED_GPUS[gpu]
            lets.append(f'if (arch == {arch}) return {i}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)

    @property
    def copy_perf_fields_body(self):
        lets = []
        for field in self.perf_fields:
            lets.append(f'param.{field} = {field}')
        ALIGN = ';\n' + ' ' * 4
        return ALIGN.join(lets)

    def incbin_mangle(self, arch, o):
        return f'INCBIN_{arch}_{self.KERNEL_FAMILY}_{self.SHIM_KERNEL_NAME}_{o.c_identifier_signature}'

    def get_kernel_table_entries(self, object_files):
        lets = []
        for gpu_index, gpu in enumerate(self._target_gpus):
            arch = AOTRITON_SUPPORTED_GPUS[gpu]
            lets.append(f'[{gpu_index}] = ' + self.get_kernel_table_entries_per_arch(arch, object_files))
        ALIGN = ',\n' + 12 * ' '
        return ALIGN.join(lets)

    def get_kernel_table_entries_per_arch(self, arch, object_files):
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
