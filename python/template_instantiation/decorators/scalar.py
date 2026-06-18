# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `ati.scalar` argument-binding decorator + its ScalarSpec record.

A ScalarSpec carries the authoring data for one non-tensor argument: a plain runtime
scalar, an enumerated choice (options=), or a member of a shared choice variable.
"""

from .choicevar import ChoiceVar
from ..specs.base import StackedSpec


class ScalarSpec(StackedSpec):
    """One non-tensor argument binding (one `@ati.scalar`).

    The first argument is either a single name or a LIST of names that share one
    choice dimension (the scalar analogue of grouping several tensors under one
    type_var; replaces the legacy `frozenset([...])` keys):

      scalar('Sm_scale', 'fp32')              -> plain runtime scalar
      scalar('CAUSAL_TYPE', options=[..])     -> enumerated (former feature)
      scalar(['Q_descale','K_descale'], options=[0])  -> several args, one axis
      scalar('Max_seqlen_q', S)               -> bound to a shared ChoiceVar S
      scalar('Num_head_q')                    -> type read from annotation (Step 2.3)
    """

    __slots__ = ('arg_names', 'type_', 'dtype', 'options', 'wires_to')

    def __init__(self, arg_name, type_or_var=None, options=None, wires_to=None):
        if isinstance(arg_name, (list, tuple)):
            assert arg_name, '@ati.scalar needs at least one argument name'
            assert all(isinstance(a, str) for a in arg_name), \
                '@ati.scalar name list must contain strings'
            assert wires_to is None, (
                '@ati.scalar with a name list cannot take wires_to= (wiring is '
                'per single argument); declare wired scalars individually')
            self.arg_names = tuple(arg_name)
        else:
            assert isinstance(arg_name, str), \
                f'@ati.scalar first arg must be a name or list of names, got {arg_name!r}'
            self.arg_names = (arg_name,)
        self.type_ = None
        self.dtype = None
        self.options = None
        # real->apparel wiring (rev0 §4.3); see TensorSpec.wires_to.
        self.wires_to = wires_to
        if isinstance(type_or_var, ChoiceVar):
            self.dtype = type_or_var
            assert options is None, \
                'cannot pass options= when binding a shared ChoiceVar'
        elif isinstance(type_or_var, str):
            self.type_ = type_or_var
            assert options is None, \
                'cannot pass both an explicit type and options='
        elif type_or_var is not None:
            raise AssertionError(
                f'@ati.scalar second arg must be a type string or ChoiceVar, '
                f'got {type_or_var!r}')
        if options is not None:
            # A numpy array with an explicit dtype is preserved AS-IS (not listified)
            # so the builder's parse_choices -> GuessNumpy path can read its dtype to
            # fix the constexpr C width (e.g. np.array([...], np.int16) -> int16_t).
            # A plain python list is listified and width-inferred (GuessInt/GuessBool).
            self.options = list(options) if not _is_numpy_array(options) else options

    @property
    def arg_name(self) -> str:
        """The representative (first) argument name."""
        return self.arg_names[0]

    @property
    def var_name(self) -> str:
        """The choice-variable name. A shared ChoiceVar uses its name; a name list
        uses its first member; a single arg names its own anonymous variable."""
        return self.dtype.name if self.dtype is not None else self.arg_names[0]

    @property
    def has_explicit_type(self) -> bool:
        # Test-only accessor (test_decorators.py); not used by the generator, which
        # checks the type_ / dtype slots individually at each site.
        return self.type_ is not None or self.dtype is not None

    def __repr__(self):
        return (f'ScalarSpec({self.arg_names!r}, type_={self.type_!r}, '
                f'dtype={self.dtype!r}, options={self.options!r})')


def _is_numpy_array(x) -> bool:
    return type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray'


def scalar(arg_name, type_or_var=None, *, options=None, wires_to=None):
    """Bind a non-tensor argument: a plain runtime scalar, an enumerated choice
    (options=), or a member of a shared choice variable. `wires_to=` dresses this
    REAL argument as an operator operand (rev0 §4.3).

    `options=` accepts a plain python list (the C width is inferred from the values)
    OR a numpy array with an explicit dtype (the dtype fixes the width), e.g.
    `options=np.array([0, 3], np.int16)` to force int16_t regardless of the values."""
    return ScalarSpec(arg_name, type_or_var, options=options, wires_to=wires_to)
