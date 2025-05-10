# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


'''
ConditionalConstexpr
  if cond_op(if_feat, feat_value):
      use then_choice
  else:
      else_choice

Determination of C type:
  - Specified by returning a single element numpy array

NOTE: THE MEMBER VARIABLES OF THIS CLASS SHOULD BE READ-ONLY
'''

from .typed_choice import ConditionalChoice, parse_choices, log

class ConditionalConstexpr(ConditionalChoice):
    def __init__(self, if_feat, if_value, then_choice, else_choice,
                 cond_op=None):
        self._if_feat = if_feat
        if cond_op is None:
            self._if_value = if_value if isinstance(if_value, list) else [if_value]
        else:
            self._if_value = if_value
        self._then = self._parse_then(then_choice)
        self._else = self._parse_else(else_choice)
        self._cond_op = cond_op
        self._link = {}
        assert not isinstance(else_choice, list), 'Cannot use iteratable else_choice in ConditionalConstexpr'
        # if isinstance(else_choice, int):
        #     assert else_dtype is not None, (
        #         f'For integer {else_choice=}, {else_dtype=} must be specified with np.int8 etc.'
        #     )

    def _parse_then(self, choice):
        return parse_choices([choice])[0]

    def _parse_else(self, choice):
        return parse_choices([choice])[0]

    def link_deferral_target(self, tp_dict):
        self._link['if'] = tp_dict[self._if_feat]

    def resolve(self, aname, tc_dict):
        if tc_dict is None:
            return self.resolve_defalut(aname)
        def is_matching():
            # print(f'{self._if_feat=}')
            # print(f'{tc_dict=}')
            # print(f'{tc_dict[self._if_feat]=}')
            # print(f'{tc_dict[self._if_feat].get_typed_value(aname)=}')
            tc_value = tc_dict[self._if_feat].triton_compile_signature
            log(lambda : f'ConditionalConstexpr.resolve: {self._if_feat=} {tc_value=} {self._if_value=}')
            if self._cond_op is None:
                return tc_value in self._if_value
            return self._cond_op(tc_value, self._if_value)
        if is_matching():
            return self.resolve_then(aname, tc_dict)
        else:
            return self.resolve_else(aname, tc_dict)

    def resolve_defalut(self, aname):
        return self.resolve_else(aname, tc_dict=None)

    def resolve_then(self, aname, tc_dict):
        return self._then

    def resolve_else(self, aname, tc_dict):
        return self._else.resolve(aname, tc_dict)

    '''
    ConditionalConstexpr may have tensor in else clause
    '''
    def resolve_rank(self, all_names, RANKS):
        self._else.resolve_rank(all_names, RANKS)

    @property
    def itype(self):
        return self.resolve_else(aname=None, tc_dict=None).itype

    @property
    def triton_compile_signature(self):
        raise f"Unresolved ConditionalConstexpr {self=}"

    def document_conditional_value(self, bind):
        return str(self._then)

    def create_constexpr(self, value):
        return self._else.create_constexpr(value)

class ConditionalDeferredConstexpr(ConditionalConstexpr):
    def link_deferral_target(self, tp_dict):
        self._link['if'] = tp_dict[self._if_feat]
        self._link['then'] = tp_dict[self._then]

    def _parse_then(self, choice):
        return choice

    def resolve_then(self, aname, tc_dict):
        return tc_dict[self._then].resolve(self._then, tc_dict)

    # ConditionalDeferredConstexpr defaults to else branch
    # def resolve_defalut(self, aname):

    def document_conditional_value(self, bind):
        tp = self._link['then']
        # print(f'Call ConditionalDeferredConstexpr.document_conditional_value(), {tp.choices=}')
        return '/'.join([str(v) for v in tp.choices])

    def create_constexpr(self, value):
        tp = self._link['then']
        return tp.repr_choice.create_constexpr(value)

# FIXME: Need specialized for Tensor so ArgumentMetadata.param_cc_fields can work
class ConditionalDeferredElseTensor(ConditionalConstexpr):
    def link_deferral_target(self, tp_dict):
        self._link['if'] = tp_dict[self._if_feat]
        self._link['else'] = tp_dict[self._else]
        log(lambda : f"Link {self=}.{self._else=} to {self._link['else']=}")

    def _parse_else(self, choice):
        return choice

    def resolve_else(self, aname, tc_dict):
        return tc_dict[self._else] # Don't call .resolve(self._else, tc_dict)

    '''
    ConditionalDeferredElseTensor defaults to else branch
    But need special handling because only TP (stored in _link['else'] is known
    ATM for else
    '''
    def resolve_defalut(self, aname):
        return self._link['else'].repr_choice.resolve(aname, tc_dict=None)

    def resolve_rank(self, all_names, RANKS):
        log(lambda : f"CDETensor.resolve_rank {all_names=} {self._link['else']=}")
        for tc in self._link['else'].choices:
            log(lambda : f"{self._link['else']=}")
            tc.resolve_rank(all_names, RANKS)

    def document_conditional_value(self, bind):
        return 'nullptr'
