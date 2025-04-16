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

class ConditionalConstexpr(object):
    def __init__(self, feat, feat_value, constexpr_value, else_choice,
                 cond_op=None,
                 else_dtype=None):
        self._constexpr = constexpr_value
        self._cond_op = cond_op
        self._when_feat = feat
        if self._cond_op is None:
            self._when_value = feat_value if isinstance(feat_value, list) else [feat_value]
        else:
            self._when_value = feat_value
        assert not isinstance(else_choice, list), 'Cannot use iteratable else_choice in ConditionalConstexpr'
        if isinstance(else_choice, int):
            assert else_dtype is not None, (
                f'For integer {else_choice=}, {else_dtype=} must be specified with np.int8 etc.'
            )
        self._else = else_choice
        self._else_dtype = else_dtype

    def _is_matching(self, fsel_dict):
        return self._matched_result(fsel_dict[self._when_feat])

    def _matched_result(self, fsel_value):
        if self._cond_op is None:
            return fsel_value in self._when_value
        return self._cond_op(fsel_value, self._when_value)

    def __call__(self, arch, fsel_dict):
        if self._is_matching(fsel_dict):
            return self.get_constexpr(fsel_dict)
        else:
            return self.get_else(fsel_dict)

    def get_constexpr(self, fsel_dict):
        return self._constexpr

    def get_else(self, fsel_dict):
        return self._else

    def get_triton_type(self):
        return self._else_dtype if self._else_dtype is not None else self._else

    @property
    def is_tensor(self):
        return isinstance(self._else, str) and self._else.startswith('*')

class ConditionalDeferredConstexpr(ConditionalConstexpr):
    def get_constexpr(self, fsel_dict):
        # print(f'{self._when_feat=} {self._constexpr=}')
        return fsel_dict[self._constexpr]

    def list_possible_constexpr_values(self, selections):
        for sel in selections:
            if sel.meta.has_argument(self._constexpr):
                return [str(v) for v in sel.meta._possible_values]

# FIXME: Need specialized for Tensor so ArgumentMetadata.param_cc_fields can work
class ConditionalDeferredElseTensor(ConditionalConstexpr):
    def get_else(self, fsel_dict):
        return fsel_dict[self._else]

    @property
    def is_tensor(self):
        return True

    def list_possible_constexpr_values(self, selections):
        return ['nullptr']
