import numpy as np
import itertools
from enum import Enum
from pathlib import Path
from .kernel_signature import KernelSignature
from .object_desc import ObjectFileDescription

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

# Use to convert dict[frozenset, str] to list[tuple(frozenset, str, int)],
# while the order present in the list is maintained by list (commonly ARGUMENTS)
def dict_to_list(d : 'dict[frozenset, tuple(str, int)]', order : list):
    pass  # TODO

'''
Note: we category the Triton kernel arguments into three types.
'''
class ArgumentCategory(Enum):
    CAT_TYPE = 1  # This argument may have differnet types
    CAT_FEAT = 2  # This argument enable/disable a specific feature
    CAT_PERF = 3  # This argument is for performance tuning

class ArgumentMetadata(object):
    def __init__(self, grouped_arguments_as_set, possible_values, cat : ArgumentCategory):
        self._grouped_arguments_as_set = grouped_arguments_as_set
        self._possible_values = possible_values
        self._npossible = len(possible_values)
        self._cat = cat
        self._godel_number = None

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
        nchoices = np.array([m.nchoices for m in sorted_metadata])
        # Big Endian, because types are usually put at first and they should be
        # considered as more important
        cumprod = np.flip(np.cumprod(np.flip(nchoices)))
        for m, c in zip(sorted_metadata, cumprod):
            m._godel_number = c

    @property
    def nchoices(self):
        return self._npossible

    def select(self, index):
        return self._possible_values[index]

    def spawn_all_selections(self):
        return [ArgumentSelection(self, i) for i in range(self.nchoices)]


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

    '''
    @property
    def FUNCTIONAL_ONLY_CHOICES(self):
        if self._FUNCTIONAL_ONLY_CHOICES is None:
            self._FUNCTIONAL_ONLY_CHOICES = join_dicts([self.TYPE_CHOICES, self.FEAT_CHOICES])
        return self._FUNCTIONAL_ONLY_CHOICES
        self.ARGUMENT_CATEGORY = {}
        for t,d in zip([ArgumentCategory.CAT_TYPE, ArgumentCategory.CAT_FEAT, ArgumentCategory.CAT_PERF],
                       [self.TYPE_CHOICES, self.FEAT_CHOICES, self.PERF_CHOICES]):
            for fs in d.keys():
                for aname in fs:
                    self.ARGUMENT_CATEGORY[aname] = t
    '''

    def __init__(self, triton_kernel_name, triton_file_path):
        self._triton_file_path = Path(triton_file_path)
        self._triton_kernel_name = triton_kernel_name
        self._func_meta = []
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_TYPE) for k, v in self.TYPE_CHOICES.items()]
        self._func_meta += [ArgumentMetadata(k, v, ArgumentCategory.CAT_FEAT) for k, v in self.FEAT_CHOICES.items()]
        self._perf_meta = [ArgumentMetadata(k, v, ArgumentCategory.CAT_PERF) for k, v in self.PERF_CHOICES.items()]
        for m in self._func_meta:
            m.sort_arguments(self.ARGUMENTS)
        self._func_meta = sorted(self._func_meta, key=lambda m: m.first_apperance)
        # print(f'{self._func_meta}')
        ArgumentMetadata.assign_godel_number(self._func_meta)
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
