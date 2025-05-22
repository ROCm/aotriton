# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from . import typed_choice as TC
from ..utils import log
'''
Bind: assocation between template parameter and specific choice (typed)

Note: Bind itself does not determine the full datatype of arguments. With C++ as analogy:

typename<typename T>
void func(Eigen::Tensor<T, 2>& t1, Eigen::Tensor<T, 3>& t2);

Bind object describes assigning T to some datatype, the complete datatype of t1 and
t2 need to be derived from info stored in Parameter object.
'''

class Bind(object):
    def __init__(self,
                 klass : 'Parameter',
                 value : 'Any',
                 nth_choice : int):
        self._klass = klass
        self._value = value
        self._conditional = isinstance(value, TC.ConditionalChoice)
        self._init_value = value
        self._nth_choice = nth_choice
        # print(f'Create Bind for {self.name}')

    ############## metadata/name/values ##############
    @property
    def param_klass(self):
        return self._klass

    @property
    def name(self):
        return self._klass.repr_name

    @property
    def value(self):
        return self._value

    def __iter__(self):
        for aname in self._klass.all_names:
            yield aname, self.get_typed_value(aname)

    def get_typed_value(self, aname) -> TC.TypedChoice:
        # print(f'get_typed_value {aname=} from Bind({self.name})')
        return self._value.resolve(aname, tc_dict=None)

    ############## signature ##############
    @property
    def godel_number(self):
        return self._nth_choice * self.param_klass.godel_number

    @property
    def show_in_compact(self):
        return self._klass.nchoices > 1

    @property
    def compact_signature(self):
        return str(self.value.triton_compile_signature) if self.show_in_compact else None

    @property
    def signature_in_func_name(self):
        if not self.show_in_compact:
            return None
        s = self.compact_signature
        s = s.replace('*', '＊').replace(':', '@').replace('True', 'T').replace('False', 'F')
        return s

    @property
    def human_readable_signature(self):
        if not self.value.HIDDEN:
            return f'{self.name} = {self.value}'
        else:
            return None

    ############## conditional ##############
    @property
    def param_maybe_conditional(self):
        return self._klass.maybe_conditional

    @property
    def is_conditional(self):
        return self._conditional

    @property
    def is_unresolved(self):
        return self.is_conditional and isinstance(self._value, TC.ConditionalChoice)

    def settle_unresolved(self, tc_dict):
        if self.is_unresolved:
            tc = self._value.resolve(self.name, tc_dict)
            log(lambda : f'settle_unresolved Bind({self.name}) from {self._value=} into {tc=}')
            setattr(self, '_value', tc)

    def document_conditional_value(self):
        assert self.is_conditional
        return self._init_value.document_conditional_value(self)


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
