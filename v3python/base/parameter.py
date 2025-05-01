# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
Purpose of Classes:

TemplateParameter: hold metadata of parameter
TypedChoice: hold choice values along with their metadata
Bind: describe the association between TemplateParameter and TypedChoice

'''

import numpy as np
from .bind import Bind
from .ttype import TI, ConditionalValue
from .typed_choice import parse_choices

class TemplateParameter(object):

    def __init__(self,
                 names,
                 choices):
        self._names = names
        self._choices = parse_choices(choices)
        self._type_dict = {}
        self._type_fallback = None

    def __sort_arguments(self, ALL_ARGUMENTS):
        arguments_tuple = [(aname, ALL_ARGUMENTS.index(aname)) for aname in self._names]
        ordered_arguments = sorted(arguments_tuple, key=lambda at: at[1])
        self._names = [ aname for aname, _ in ordered_arguments ]
        self._first_apperance = ordered_arguments[0][1]

    def late_init(self, ALL_ARGUMENTS, param_dict, RANKS, STRIDES):
        self.__sort_arguments(ALL_ARGUMENTS)
        # Assocate ConditionalValue with the depending values
        for c in self._choices:
            if not self.is_conditional_value(c):
                continue
            self._maybe_conditional = True
            c.link_deferral_target(param_dict)
        from .guesstype import guess_type
        ttype = guess_ttype(self._choices)
        if ttype.is_tensor:
            self._type_dict = { aname : parse_tensor_type(aname, self.nchoices, RANKS) for aname in self.all_names ]

    @property
    def repr_name(self):
        return self._names[0]

    @property
    def first_apperance(self):
        return self._first_apperance

    @property
    def all_names(self):
        return self._names

    @property
    def choices(self):
        return self._choices

    @property
    def nchoices(self):
        return len(self._choices)

    def create_nth(self, nth):
        return Bind(self, self._choices[nth], nth)

    def create_direct(self, value):
        return Bind(self, value, None)

    def __iter__(self):
        for nth, v in enumerate(self._choices):
            yield Bind(self, v, nth)

    @property
    def maybe_conditional(self):
        return self._maybe_conditional

    def is_conditional_value(self, value):
        return isinstance(value, ConditionalValue)

    @property
    def ttype(self):
        return self._ttype

    @property
    def godel_number(self):
        return self._godel_number

    @staticmethod
    def assign_godel_number(sorted_parameters : list):
        # No need to trim m.nchoices == 1 because they have no impact on the assignment
        # However we need to append 1 as the initial "stride"
        nchoices = np.array([m.nchoices for m in sorted_parameters] + [1])
        # Big Endian with two np.flip() calls, because types are usually put at first
        # and they should be considered as more important.
        # The first stride is trimmed off to ensure last stride is 1
        cumprod = np.flip(np.cumprod(np.flip(nchoices)))[1:]
        for m, c in zip(sorted_parameters, cumprod):
            m._godel_number = c

class TypeParameter(TemplateParameter):
    @property
    def field_ctype(self):
        if self.nchoices == 1:
            assert isinstance(self._choices[0], TI)
            return self._choices[0].ctype
        assert all([c.is_tensor for c in self._choices])
        return self._choices[0].ctype

class ValueParameter(TemplateParameter):
    @property
    def field_ctype(self):
        return self._ttype.ctype

'''
all names -> arg dict
'''
def build_complete_mdict(params):
    return { aname : param for param in params for aname in param.all_names }
