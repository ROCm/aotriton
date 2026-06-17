# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Argument-binding decorators: `ati.tensor` / `ati.scalar` and their spec records.

A TensorSpec/ScalarSpec carries the authoring data for one argument plus, for tensors,
the logic to resolve stride globs and infer rank against a concrete parameter-name list.
"""

import fnmatch

from .choicevar import ChoiceVar


class TensorSpec:
    """One tensor argument binding (one `@ati.tensor`).

    The first argument is either a single name or a LIST of names that share one
    dtype and shape but have NO strides — e.g. the rank-0 philox pointers
    (replaces grouping them under one frozenset). Strides require a per-tensor
    glob, so a name list forbids `strides=`/`contiguous=`.

    `dtype` is either a ChoiceVar (shared dimension) or a literal type string
    (anonymous single-choice variable). Strides are bound per tensor via a glob;
    rank is inferred from the matched stride count unless given explicitly."""

    __slots__ = ('arg_names', 'dtype', 'strides_pattern', 'rank', 'contiguous',
                 'wires_to', 'resolved_strides')

    def __init__(self, arg_name, dtype, strides=None, rank=None, contiguous=None,
                 wires_to=None, resolved_strides=None):
        assert isinstance(dtype, (ChoiceVar, str)), \
            f'@ati.tensor dtype must be a ChoiceVar or a literal type string, ' \
            f'got {dtype!r}'
        if isinstance(arg_name, (list, tuple)):
            assert arg_name, '@ati.tensor needs at least one argument name'
            assert all(isinstance(a, str) for a in arg_name), \
                '@ati.tensor name list must contain strings'
            assert strides is None and contiguous is None, (
                '@ati.tensor with a name list cannot take strides=/contiguous= '
                '(a stride glob cannot be partitioned across several tensors); '
                'group only strideless tensors, or declare them individually')
            assert wires_to is None, (
                '@ati.tensor with a name list cannot take wires_to= (wiring is '
                'per single argument); declare wired tensors individually')
            self.arg_names = tuple(arg_name)
        else:
            assert isinstance(arg_name, str), \
                f'@ati.tensor first arg must be a name or list of names, got {arg_name!r}'
            self.arg_names = (arg_name,)
        self.dtype = dtype
        self.strides_pattern = strides
        self.rank = rank
        self.contiguous = contiguous          # int index | stride-name | None
        # real->apparel wiring (rev0 §4.3): the operator operand this REAL argument
        # is dressed as. A plain operand-name string today; representation kept
        # opaque for the future tuple[Callable, list[str]] reducer form. None =
        # not wired (identity).
        self.wires_to = wires_to
        # The EXACT stride argument names this tensor binds, in order. Normally
        # None (derived from `strides`/the glob against the kernel signature). When
        # a strided tensor is inherited through @ati.cite, the resolver copies the
        # cited tensor's RESOLVED stride names here so the citing kernel binds the
        # exact same argument names — never re-globbing, which could match a
        # different set if the two kernels name their strides differently
        # (stride_a0/a1 vs stride_az/ah). match_strides() returns these verbatim
        # when present.
        self.resolved_strides = list(resolved_strides) if resolved_strides else None

    @property
    def arg_name(self) -> str:
        """The representative (first) argument name."""
        return self.arg_names[0]

    @property
    def is_literal_dtype(self) -> bool:
        return isinstance(self.dtype, str)

    @property
    def var_name(self) -> str:
        """The choice-variable name this tensor binds. A literal dtype is sugar
        for an anonymous single-choice variable named after the first argument."""
        return self.dtype.name if isinstance(self.dtype, ChoiceVar) else self.arg_names[0]

    def match_strides(self, param_names) -> list[str]:
        """The tensor's stride parameter names, in order. If exact resolved names
        were recorded (the @ati.cite path), return those verbatim (asserting they
        are all in the signature) — never re-glob, since the citing kernel may name
        its strides differently. Otherwise glob-match `strides_pattern` against the
        signature."""
        if self.resolved_strides is not None:
            missing = [s for s in self.resolved_strides if s not in param_names]
            assert not missing, (
                f'@ati.tensor({self.arg_name!r}) cited stride names {missing} are '
                f'not in the kernel signature {list(param_names)}')
            return list(self.resolved_strides)
        if self.strides_pattern is None:
            return []
        return [p for p in param_names
                if fnmatch.fnmatchcase(p, self.strides_pattern)]

    def resolve_rank(self, param_names) -> int:
        """Explicit rank wins; otherwise infer from the matched stride count."""
        if self.rank is not None:
            return self.rank
        strides = self.match_strides(param_names)
        assert strides, (
            f'@ati.tensor({self.arg_name!r}) has neither rank= nor a matching '
            f'strides= pattern; cannot infer rank')
        return len(strides)

    def resolve_contiguous(self, param_names) -> str | None:
        """Resolve `contiguous=` to a concrete stride parameter name.
        An int indexes this tensor's matched stride list (negative ok); a string
        is taken as the stride name directly; None means no unit stride."""
        if self.contiguous is None:
            return None
        if isinstance(self.contiguous, int):
            strides = self.match_strides(param_names)
            assert strides, (
                f'@ati.tensor({self.arg_name!r}) contiguous={self.contiguous} '
                f'needs a strides= pattern to index into')
            return strides[self.contiguous]       # negative index = Python rule
        assert isinstance(self.contiguous, str)
        return self.contiguous

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this spec onto the kernel below it."""
        from ..describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return (f'TensorSpec({self.arg_name!r}, dtype={self.dtype!r}, '
                f'strides={self.strides_pattern!r}, rank={self.rank}, '
                f'contiguous={self.contiguous!r})')


class ScalarSpec:
    """One non-tensor argument binding (one `@ati.scalar`).

    The first argument is either a single name or a LIST of names that share one
    choice dimension (the scalar analogue of grouping several tensors under one
    tensor_dtype; replaces the legacy `frozenset([...])` keys):

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
        return self.type_ is not None or self.dtype is not None

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this spec onto the kernel below it."""
        from ..describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return (f'ScalarSpec({self.arg_names!r}, type_={self.type_!r}, '
                f'dtype={self.dtype!r}, options={self.options!r})')


def _is_numpy_array(x) -> bool:
    return type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray'


def tensor(arg_name, dtype, *, strides=None, rank=None, contiguous=None,
           wires_to=None):
    """Bind a tensor argument to a dtype variable (or literal type string).
    `wires_to=` dresses this REAL argument as an operator operand (rev0 §4.3)."""
    return TensorSpec(arg_name, dtype, strides=strides, rank=rank,
                      contiguous=contiguous, wires_to=wires_to)


def scalar(arg_name, type_or_var=None, *, options=None, wires_to=None):
    """Bind a non-tensor argument: a plain runtime scalar, an enumerated choice
    (options=), or a member of a shared choice variable. `wires_to=` dresses this
    REAL argument as an operator operand (rev0 §4.3)."""
    return ScalarSpec(arg_name, type_or_var, options=options, wires_to=wires_to)
