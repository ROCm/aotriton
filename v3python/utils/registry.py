# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from collections import defaultdict

class StringRegistry(object):
    def __init__(self):
        self._string_dict = {None:0}

    def register(self, s):
        assert s is not None # string_dict[None] tracks the total size
        if s in self._string_dict:
            return self._string_dict[s]
        offset = self._string_dict[None]
        self._string_dict[s] = offset
        self._string_dict[None] = offset + len(s) + 1  # Need a trailing '\0'
        return offset

    '''
    NOTE: DO NOT MOVE THE C CODE CONSTRUCTION TO CODEGEN PACKAGE.
          Unlike other registry, string registry's C code is tightly coupled
          with the returned offset values.
    '''
    def get_data(self):
        packed_string = '\n'.join(['"' + s + '\\0"' for s in self._string_dict if s is not None])
        return packed_string

@dataclass
class FunctionItem:
    ret    : str = ''
    name   : str = ''
    params : str = ''

class FunctionRegistry(object):
    def __init__(self):
        self._function_registry = {}

    def register(self, fsrc, fret, fname_pfx, fparams):
        if fsrc in self._function_registry:
            return self._function_registry[fsrc].name
        findex = len(self._function_registry)
        fname = fname_pfx + f'__{findex}'
        self._function_registry[fsrc] = FunctionItem(ret=fret, name=fname, params=fparams)
        return fname

    def get_data(self):
        return self._function_registry

class SignaturedFunctionRegistry(object):
    def __init__(self):
        self._function_registry = {}

    def register(self, fsig, fsrc):
        if fsig in self._function_registry:
            return self._function_registry[fsig][0]
        findex = len(self._function_registry)
        self._function_registry[fsig] = (findex, fsrc)
        return findex

    def contains(self, fsig):
        if fsig in self._function_registry:
            return True, self._function_registry[fsig][0]
        return False, None

    def get_data(self):
        return self._function_registry

class HsacoRegistry(object):
    def __init__(self):
        self._rule_registry = defaultdict(list)

    def register(self, functional, signatures, *, append=False):
        if append:
            self._rule_registry[functional].append(signatures)
        else:
            self._rule_registry[functional] = signatures

    def get_data(self):
        return self._rule_registry

class ListRegistry(object):
    def __init__(self):
        self._list_registry = []

    def register(self, elem):
        self._list_registry.append(elem)

    def get_data(self):
        return self._list_registry

class DictRegistry(object):
    def __init__(self):
        self._dict_registry = {}

    def register(self, key, value):
        self._dict_registry[key] = value

    def get_data(self):
        return self._dict_registry

'''
Class to register re-used code or objects
'''

class RegistryRepository(object):
    def __init__(self):
        self._subreg_dict = {}

    def _get_registry_with_factory(self, name, factory):
        if name not in self._subreg_dict:
            self._subreg_dict[name] = factory()
        return self._subreg_dict[name]

    def get_string_registry(self, name):
        return self._get_registry_with_factory(name, StringRegistry)

    def get_function_registry(self, name):
        return self._get_registry_with_factory(name, FunctionRegistry)

    def get_signatured_function_registry(self, name):
        return self._get_registry_with_factory(name, SignaturedFunctionRegistry)

    def get_hsaco_registry(self, name):
        return self._get_registry_with_factory(name, HsacoRegistry)

    def get_list_registry(self, name):
        return self._get_registry_with_factory(name, ListRegistry)

    def get_dict_registry(self, name):
        return self._get_registry_with_factory(name, DictRegistry)

    def get_data(self, name, return_none=False):
        if not return_none:
            return self._subreg_dict[name].get_data()
        if name in self._subreg_dict:
            return self._subreg_dict[name].get_data()
        return None
