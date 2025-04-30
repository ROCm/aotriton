# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from .conditional_value import ConditionalConstexpr
from .ttype import guess_type

class Parameter(object):

    def __init__(self,
                 names,
                 choices,
                 ttype=None):
        self._names = names
        self._choices = choices
        self._maybe_conditional = any([self.is_conditional_value(c) for c in choices])
        self._ttype = ttype

    def sort_arguments(self, ALL_ARGUMENTS):
        arguments_tuple = [(aname, ALL_ARGUMENTS.index(aname)) for aname in self._names]
        ordered_arguments = sorted(arguments_tuple, key=lambda at: at[1])
        self._names = [ aname for aname, _ in ordered_arguments ]

    @property
    def repr_name(self):
        return self._names[0]

    @property
    def all_names(self):
        return self._names

    @property
    def choices(self):
        return self._choices

    def create_nth(self, nth):
        return Argument(self, self._choices[nth], nth)

    def create_direct(self, value):
        return Argument(self, value, None)

    @property
    def maybe_conditional(self):
        return self._maybe_conditional

    def is_conditional_value(self, value):
        return isinstance(value, ConditionalConstexpr)

class TypeParameter(Parameter):
    pass

class ValueParameter(Parameter):
    pass

class Argument(object):
    def __init__(self,
                 klass : Parameter,
                 value : 'Any',
                 nth_choice : int):
        self._klass = klass
        self._value = value
        if self._klass.is_conditional_value(value):
            self._conditional = value
        self._nth_choice = nth_choice

    @property
    def param_klass(self):
        return self._klass

    @property
    def name(self):
        return self._klass.repr_name

    @property
    def value(self):
        return self._value

    @property
    def cvalue(self) -> str:
        return self._klass._ttype.format_cvalue(self.value)

    @property
    def maybe_conditional(self):
        return self._klass.maybe_conditional

    @property
    def is_unresolved(self):
        return self._klass.maybe_conditional and self._klass.is_conditional_value(self._value)

    @property
    def settle_unresolved(self, arch, sel_dict):
        self._value = self._value(arch, sel_dict)

    @property
    def possible_constexpr_values(self):
        return self._conditional.list_possible_constexpr_values(self._klass)

    def format_constexpr(self):
        return self._conditional.format_constexpr(self)

'''
repr_name -> value dict
'''
def build_dict(args):
    return { arg.name : arg.value for arg in args }

'''
all names -> arg dict
'''
def build_complete_dict(args):
    return { aname : arg for arg in args for aname in arg._klass.all_names }

'''
repr_name -> value dict
Only contain items with multiple selections
'''
def build_compact_dict(args):
    return { arg.name : arg.value for arg in args if arg.show_in_compact }
