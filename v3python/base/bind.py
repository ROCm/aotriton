# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
        if self._klass.is_conditional_value(value):
            self._conditional = value
        self._nth_choice = nth_choice

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

    @property
    def cvalue(self) -> str:
        return self._klass._ttype.format_cvalue(self.value)

    ############## signature ##############
    @property
    def godel_number(self):
        return self._nth_choice * self.param_klass.godel_number

    @property
    def show_in_compact(self):
        return self._klass.nchoices > 1

    @property
    def compact_signature(self):
        return str(self.value) if self.show_in_compact else None

    @property
    def human_readable_signature(self):
        return f'{self.name} = {self.value}'

    ############## conditional ##############
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
