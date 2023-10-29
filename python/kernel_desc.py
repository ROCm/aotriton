#!/usr/bin/env python

from abc import ABC, abstractmethod
from pathlib import Path

from object_desc import ObjectFileDescription

def get_possible_types(klass, arg_name : str) -> list[str]:
    for k, v in klass.ARGUMENT_CHOICES.items():
        if arg_name in k:
            return v
    assert False, f"cannot find {arg_name}"


class KernelDescription(ABC):
    ARGUMENTS = []
    ARGUMENT_CHOICES = {}
    SHIM_KERNEL_NAME = None

    def __init__(self, triton_kernel_name, triton_file_path):
        self._triton_file_path = Path(triton_file_path)
        self._triton_kernel_name = triton_kernel_name

    def _recursive_choice(self, keys, index):
        if index >= len(keys):
            yield {}
            return
        key = keys[index]
        for choice in self.ARGUMENT_CHOICES[key]:
            for child_choice in self._recursive_choice(keys, index+1):
                child_choice[key] = choice
                yield child_choice

    def gen_all_possible_choices(self):
        keys = list(self.ARGUMENT_CHOICES.keys())
        yield from self._recursive_choice(keys, 0)

    def make_argument_choice(self, choice):
        ret = list(self.ARGUMENTS)
        for i in range(len(ret)):
            for k, v in choice.items():
                if ret[i] in k:
                    ret[i] = v
        return ret

    @property
    def all_possible_signatures(self) -> list[list[str]]:
        # cartesian = [self.argument_choices[aname] for aname in self.arguments]
        # return itertools.product(cartesian)
        all_sigs = []
        for choice in self.gen_all_possible_choices():
            all_sigs.append(self.make_argument_choice(choice))
        return all_sigs

    # Use list to maintain the order
    @property
    def arguments(self) -> list[str]:
        return self.ARGUMENTS

    # Use dict to make lookup easier
    @property
    def argument_choices(self) -> list[frozenset[str], list[str]]:
        return self.ARGUMENT_CHOICES

    def get_object_files(self, outpath : Path, prefix='') -> 'ObjectFileDescription':
        ret = []
        for choice in self.gen_all_possible_choices():
            sig_list = self.make_argument_choice(choice)
            sig = self.compact_mangle(choice)
            fn = prefix + '-' + sig + '.hsaco'
            ret.append(ObjectFileDescription(self, choice, sig_list, outpath / fn))
        return ret

    def mangle(self, sig : list[str]):
        # * -> ^: Pascal Pointer
        # : -> @: A(@)lign
        mangle_sig = [ str(t).replace('*', '^').replace(':', '@') for t in sig ]
        return ','.join(mangle_sig)

    def compact_mangle(self, choice : dict[frozenset[list], str]):
        compact_sig = []
        assigned_args = set()
        for aname in self.ARGUMENTS:
            if aname in assigned_args:
                continue
            for k, v in choice.items():
                if aname in k:
                    compact_sig.append(v)
                    for other_arg in k:
                        assigned_args.add(other_arg)
                    break
        return self.mangle(compact_sig)
