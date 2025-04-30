# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
        packed_string = '\n'.join(['"' + s + '\\0"' for s in self._string_dict])
        return packed_string


class FunctionRegistry(object):
    def __init__(self):
        self._function_registry = {}

    def register(self, fsrc, fret, fname_pfx, fparams):
        if fsrc in self._function_registry:
            return self._function_registry[fsrc][0]
        findex = len(self._function_registry)
        fname = fname_pfx + f'__{findex}'
        self._function_registry[fsrc] = (fret, fname, fparams)
        return fname

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
        return fsig in self._function_registry

'''
Class to register re-used code or objects
'''

class RegistryRepository(object):
    def __init__(self):
        self._subreg_dict = {}

    def get_string_registry(name):
        if name not in self._subreg_dict:
            self._subreg_dict[name] = StringRegistry()
        return self._subreg_dict[name]

    def get_function_registry(name):
        if name not in self._subreg_dict:
            self._subreg_dict[name] = FunctionRegistry()
        return self._subreg_dict[name]

    def get_signatured_function_registry(name):
        if name not in self._subreg_dict:
            self._subreg_dict[name] = SignaturedFunctionRegistry()
        return self._subreg_dict[name]

    def get_data(name):
        return self._subreg_dict[name].get_data()
