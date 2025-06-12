# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

'''
See README.md for the design of the Parameter system
'''

import numpy as np
from .bind import Bind
# from .ttype import TI, ConditionalValue
from . import typed_choice as TC
from .cfield import cfield
from ..utils import log

class TemplateParameter(object):

    def __init__(self,
                 names,
                 choices):
        self._names = names
        self._choices = TC.parse_choices(choices)
        self._maybe_conditional = any([isinstance(c, TC.ConditionalChoice) for c in self._choices])
        self._type_dict = {}
        self._type_fallback = None
        log(lambda : f'TP {self._names=} Done')

    def __sort_arguments(self, ALL_ARGUMENTS):
        arguments_tuple = [(aname, ALL_ARGUMENTS.index(aname)) for aname in self._names]
        ordered_arguments = sorted(arguments_tuple, key=lambda at: at[1])
        self._names = [ aname for aname, _ in ordered_arguments ]
        # alocs = "Argument LOCationS"
        self._first_apperance = ordered_arguments[0][1]
        self._ordered_arguments = ordered_arguments

    def late_init(self, ALL_ARGUMENTS, tp_dict, RANKS, STRIDES):
        log(lambda : f'TP {self._names=} late_init Start')
        self.__sort_arguments(ALL_ARGUMENTS)
        # Assocate ConditionalValue with the depending values
        for c in self._choices:
            if not isinstance(c, TC.ConditionalChoice):
                continue
            # TODO: Do we really need this?
            c.link_deferral_target(tp_dict)
        for tc in self._choices:
            tc.resolve_rank(self.all_names, RANKS)
        log(lambda : f'TP {self._names=} late_init Done')

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
    def repr_choice(self):
        return self._choices[0]

    @property
    def repr_typed_choice(self):
        return self.repr_choice.resolve(self.repr_name, tc_dict=None)

    @property
    def choices(self):
        return self._choices

    @property
    def nchoices(self):
        return len(self._choices)

    def create_nth(self, nth):
        return Bind(self, self._choices[nth], nth)

    '''
    Do not yield Bind directly
    itertools.product will re-use "outer" bind objects.
    However, in-order to support ConditionalChoice, bind objects store the
    settled values for Functionals and thus cannot be re-used.
    '''
    def __iter__(self):
        for nth in range(self.nchoices):
            yield nth

    @property
    def maybe_conditional(self):
        return self._maybe_conditional

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

    def get_cfields(self):
        def _gen():
            for aname, index in self._ordered_arguments:
                resolved_tc = self.repr_choice.resolve(aname, tc_dict=None)
                if resolved_tc.HIDDEN:
                    # print(f'HIDDEN {aname=} {resolved_tc=}')
                    continue
                yield cfield(ctype=resolved_tc.itype,
                             aname=aname,
                             index=index,
                             nbits=resolved_tc.NBITS)
        return list(_gen())

    def get_unprocessed_cfields(self):
        def _gen():
            for index, aname in enumerate(self._names):
                resolved_tc = self.repr_choice.resolve(aname, tc_dict=None)
                yield cfield(ctype=resolved_tc.itype,
                             aname=aname,
                             index=index,
                             nbits=resolved_tc.NBITS)
        return list(_gen())

# class TypeParameter(TemplateParameter):
#     @property
#     def field_ctype(self):
#         if self.nchoices == 1:
#             assert isinstance(self._choices[0], TI)
#             return self._choices[0].ctype
#         assert all([c.is_tensor for c in self._choices])
#         return self._choices[0].ctype

class PerformanceTemplateParameter(TemplateParameter):
    # @property
    # def field_ctype(self):
    #     return self._ttype.ctype
    '''
    create_direct is only sensible for performance options.
    For Functional TP, create_direct will nullify it godel number.

    So, move this function to the subclass as built-in sanity check.
    '''
    def create_direct(self, value):
        # assert isinstance(self.repr_choice, TC.constexpr_base), f'{self.repr_choice.__class__=} is not subclass of TC.constexpr_base'
        log(lambda : f'create_direct {self.repr_name=} {value=} {self.repr_choice=}')
        typed_value = self.repr_choice.create_constexpr(value)
        return Bind(self, typed_value, None)


'''
all names -> arg dict
'''
def build_complete_mdict(params):
    return { aname : param for param in params for aname in param.all_names }
