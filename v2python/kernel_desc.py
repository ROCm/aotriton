import numpy as np
import itertools
from collections import defaultdict
import io
from enum import Enum
from pathlib import Path
from .kernel_signature import KernelSignature
from .object_desc import ObjectFileDescription

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

'''
Note: we category the Triton kernel arguments into three types.
'''
class ArgumentCategory(Enum):
    CAT_TYPE = 1  # This argument may have differnet types
    CAT_FEAT = 2  # This argument enable/disable a specific feature
    CAT_PERF = 3  # This argument is for performance tuning

class ArgumentMetadata(object):
    DTYPE_NUMBER = {
        'fp16' : 'DType::kFloat16',
        'bf16' : 'DType::kBFloat16',
        'fp32' : 'DType::kFloat32',
    }
    def __init__(self, grouped_arguments_as_set, possible_values, cat : ArgumentCategory, kdesc):
        self._grouped_arguments_as_set = grouped_arguments_as_set
        self._possible_values = possible_values
        self._npossible = len(possible_values)
        self._cat = cat
        self._godel_number = None
        self._kdesc = kdesc

    def sort_arguments(self, ALL_ARGUMENTS):
        arguments_tuple = [(a, ALL_ARGUMENTS.index(a)) for a in self._grouped_arguments_as_set]
        self._ordered_arguments = sorted(arguments_tuple, key=lambda at: at[1])
        self._first_apperance = self._ordered_arguments[0][1]

    @property
    def first_apperance(self):
        assert hasattr(self, '_first_apperance'), 'ArgumentMetadata: must call sort_arguments before first_apperance'
        return self._first_apperance

    @property
    def ordered_argument_places(self):
        assert hasattr(self, '_ordered_arguments'), 'ArgumentMetadata: must call sort_arguments before ordered_argument_places'
        return [a[1] for a in self._ordered_arguments]

    @property
    def godel_number(self):
        assert hasattr(self, '_godel_number'), 'ArgumentMetadata: must call assign_godel_number on all ArgumentMetadata objects before using godel_number property'
        return self._godel_number

    @staticmethod
    def assign_godel_number(all_metadata: 'list[ArgumentMetadata]'):
        sorted_metadata = sorted(all_metadata, key=lambda m: m.first_apperance)
        for i, m in enumerate(sorted_metadata):
            m._order_among_all_choices = i
        # No need to trim m.nchoices == 1 because they have no impact on the assignment
        # (Analog: size 1 dimensions in Tensor)
        # However we need to append 1 as the last "stride"
        nchoices = np.array([m.nchoices for m in sorted_metadata] + [1])
        # Big Endian, because types are usually put at first and they should be
        # considered as more important
        # The first stride is trimmed off to ensure last stride is 1
        cumprod = np.flip(np.cumprod(np.flip(nchoices)))[1:]
        for m, c in zip(sorted_metadata, cumprod):
            m._godel_number = c

    @property
    def nchoices(self):
        return self._npossible

    def select(self, index):
        return self._possible_values[index]

    def spawn_all_selections(self):
        return [ArgumentSelection(self, i) for i in range(self.nchoices)]

    @property
    def is_tensor(self):
        triton_type = self._possible_values[0]
        return isinstance(triton_type, str) and triton_type.startswith('*')

    @property
    def cc_type(self):
        triton_arg = self._ordered_arguments[0][0]
        triton_type = self._possible_values[0]
        if self.is_tensor:
            rank = self._kdesc.get_tensor_rank(triton_arg)
            return f'T{rank}'
        if isinstance(triton_type, str):
            return ObjectFileDescription.SIGNATURE_TO_C[triton_type]
        elif isinstance(triton_type, bool):
            return 'bool'
        elif isinstance(triton_type, int):
            return 'int32_t'
        elif isinstance(triton_type, float):
            return 'float'
        assert False, f'{triton_arg} {triton_type}'

    @property
    def param_cc_fields(self):
        triton_arg = self._ordered_arguments[0][0]
        if triton_arg.startswith('stride_'):
            return []
        cc_type = self.cc_type
        return [ cc_type + ' ' + a[0] for a in self._ordered_arguments ]
        # ret = [ cc_type + ' ' + a[0] for a in self._ordered_arguments ]
        # print(f'{ret=}')

    def write_godel_number_calculation(self, fout):
        if self.nchoices <= 1:
            return
        triton_arg = self._ordered_arguments[0][0]
        triton_type = self._possible_values[0]
        INDENT = 4 * ' '
        print(INDENT + '{', file=fout)
        print(2 * INDENT + 'int64_t number = 0;', file=fout)
        if self.is_tensor:
            for number, possible_type in enumerate(self._possible_values):
                elem_type = possible_type[1:].split(':')[0]
                print(2 * INDENT + f'if ({triton_arg}.dtype() == {self.DTYPE_NUMBER[elem_type]}) number = {number} ;', file=fout)
        else:
            for number, possible_type in enumerate(self._possible_values):
                value = str(possible_type).lower()
                print(2 * INDENT + f'if ({triton_arg} == {value}) number = {number} ;', file=fout)
        print(2 * INDENT + f'sum += number * {self.godel_number};', file=fout)
        print(1 * INDENT + '}', file=fout)

class ArgumentSelection(object):
    def __init__(self, meta : ArgumentMetadata, selection_index : int):
        self._meta = meta
        self._selection_index = selection_index
        self._selection_value = self._meta.select(selection_index)

    @property
    def godel_number(self):
        return self._meta.godel_number * self._selection_index

    @property
    def triton_signature(self):
        return str(self._selection_value)

    @property
    def c_symbol_signature(self):
        return self.triton_signature.replace('*', '^').replace(':', '@')

    # compact_signature implies its usage: in file name and/or as C symbol
    @property
    def compact_signature(self):
        if self._meta.nchoices <= 1:
            return None
        return self.c_symbol_signature

    def update_triton_api_signature(self, sig: dict):
        for place in self._meta.ordered_argument_places:
            sig[place] = self.triton_signature

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

    def gen_func_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._func_selections)

    def gen_perf_selections(self) -> 'tuple[ArgumentSelection]':
        return itertools.product(*self._perf_selections)

    def gen_all_object_files(self, outpath : Path, file_name_prefix='') -> 'list[ObjectFileDescription]':
        for fsels in self.gen_func_selections():
            for psels in self.gen_perf_selections():
                sig = KernelSignature(self, fsels, psels)
                fn = file_name_prefix + '-' + sig.compact_signature + '.hsaco'
                yield ObjectFileDescription(self, sig, outpath / fn)

    @property
    def param_class_name(self):
        return "".join(x.capitalize() for x in self.SHIM_KERNEL_NAME.lower().split("_"))

    def write_launcher_header(self, fout):
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'param_class_name'    : self.param_class_name,
              'func_fields'         : ';\n    '.join(sum([m.param_cc_fields for m in self._func_meta], [])),
              'perf_fields'         : ';\n    '.join(sum([m.param_cc_fields for m in self._perf_meta], [])),
            }
        print(self.HEADER_TEMPLATE.format_map(d), file=fout)

    def write_launcher_source(self, fout, object_files):
        d = { 'kernel_family_name'  : self.KERNEL_FAMILY,
              'shim_kernel_name'    : self.SHIM_KERNEL_NAME,
              'param_class_name'    : self.param_class_name,
              'godel_number_body'   : self.godel_number_body,
              'let_tensor_stride_arguments' : self.let_tensor_stride_arguments,
              'let_kernel_arguments' : self.let_kernel_arguments,
              'arch_godel_number'    : self._godel_number,
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

    def get_kernel_table_entries(self, object_files):
        d = defaultdict(list)
        lets = ['{']
        for o in object_files:
            godel_number = o.godel_number
            d[godel_number].append(f'INCBIN_{self.KERNEL_FAMILY}_{self.SHIM_KERNEL_NAME}_{o.compact_signature}')
        for k, v in d.items():
            lets.append(f'[{k}] =' + '{' + ', '.join(v) + '}')
        lets.append('}')
        ALIGN = ',\n' + 20 * ' '
        return ALIGN.join(lets)
