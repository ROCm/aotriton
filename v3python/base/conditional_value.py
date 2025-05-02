# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

#
# ConditionalConstexpr
#   if cond_op(feat, feat_value):
#       use constexpr_value
#   else:
#       else_choice
#
# Determination of C type:
#   - Guessed returned value
#   - Specified by returning a single element numpy array
#

from .typed_choice import ConditionalChoice, parse_choices

class ConditionalConstexpr(ConditionalChoice):
    def __init__(self, feat, feat_value, constexpr_value, else_choice,
                 cond_op=None):
        self._when_feat = feat
        if cond_op is None:
            self._when_value = feat_value if isinstance(feat_value, list) else [feat_value]
        else:
            self._when_value = feat_value
        self._constexpr = parse_choices([constexpr_value])[0]
        self._else = parse_choices([else_choice])[0]
        self._cond_op = cond_op
        assert not isinstance(else_choice, list), 'Cannot use iteratable else_choice in ConditionalConstexpr'
        # if isinstance(else_choice, int):
        #     assert else_dtype is not None, (
        #         f'For integer {else_choice=}, {else_dtype=} must be specified with np.int8 etc.'
        #     )

    def resolve(self, aname, bind_dict):
        def is_matching():
            bond_value = bind_dict[self._when_feat].get_typed_value(aname).triton_compile_signature
            if self._cond_op is None:
                return bond_value in self._when_value
            return self._cond_op(bond_value, self._when_value)
        if not is_matching() or bind_dict is None:
            return self.resolve_else(aname, bind_dict)
        else:
            return self.resolve_then(aname, bind_dict)

    def resolve_then(self, aname, bind_dict):
        return self._constexpr

    def resolve_else(self, aname, bind_dict):
        return self._else.resolve(aname, bind_dict)

    @property
    def itype(self):
        return self.resolve_else(aname=None, bind_dict=None).itype

    @property
    def triton_compile_signature(self):
        raise f"Unresolved ConditionalConstexpr {self=}"

    def document_conditional_value(self, bind):
        return str(self._constexpr)

class ConditionalDeferredConstexpr(ConditionalConstexpr):
    def document_conditional_value(self, bind):
        tp = bind.param_klass
        return '/'.join([str(v) for v in param.choices])

# FIXME: Need specialized for Tensor so ArgumentMetadata.param_cc_fields can work
class ConditionalDeferredElseTensor(ConditionalConstexpr):

    def document_conditional_value(self, bind):
        return 'nullptr'
