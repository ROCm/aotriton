# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI choice-declaring decorators (executive plan Step 2.1; agent-plans/ati_rev1.md
§3.1–§3.2).

These produce *spec records* describing how an argument is instantiated; they are
not yet attached to a kernel (that is Step 2.3 — describe() + the @-sugar) and do
not yet build Axis/Override IR (Step 2.4). A spec carries the authoring data plus,
for tensors, the logic to resolve stride globs and infer rank against a concrete
parameter-name list.

Surface:
    T   = ati.tensor_dtype('T_io', dtype=[...])      # named dtype variable
    S   = ati.choice_set('Seqlen', options=[...])    # named scalar choice variable
    ati.tensor('Q', T, strides='stride_q?', contiguous=-1)
    ati.tensor('L', '*fp32:16', rank=2)              # literal dtype
    ati.scalar('Sm_scale', 'fp32')                   # plain runtime scalar
    ati.scalar('CAUSAL_TYPE', options=[0, 3])        # enumerated (former feature)
    ati.scalar('Max_seqlen_q', S)                    # bound to a shared variable
"""

import fnmatch

from .ir import Override, VarRef
from .ir import eq, ne, lt, gt, le, ge   # predicate builders, re-exported as ati.*


class ChoiceVar:
    """A named choice variable shared by several arguments — the
    `template<typename T>` / TypeVar analogue. `tensor_dtype` and `choice_set`
    both produce one; `kind` records which authoring word created it (purely for
    diagnostics — Choice.parse handles tensor vs scalar literals uniformly)."""

    __slots__ = ('name', 'choices', 'kind')

    def __init__(self, name, choices, kind):
        assert isinstance(name, str) and name, 'choice variable needs a name'
        self.name = name
        self.choices = list(choices)
        self.kind = kind          # 'dtype' | 'scalar'

    def __repr__(self):
        return f'ChoiceVar({self.name!r}, kind={self.kind}, choices={self.choices})'


def tensor_dtype(name, dtype):
    """Declare a named tensor element-type variable. `dtype` is the choice set."""
    return ChoiceVar(name, dtype, kind='dtype')


def choice_set(name, options):
    """Declare a named scalar choice variable. `options` is the choice set."""
    return ChoiceVar(name, options, kind='scalar')


class TensorSpec:
    """One tensor argument binding (one `@ati.tensor`).

    `dtype` is either a ChoiceVar (shared dimension) or a literal type string
    (anonymous single-choice variable). Strides are bound per tensor via a glob;
    rank is inferred from the matched stride count unless given explicitly."""

    __slots__ = ('arg_name', 'dtype', 'strides_pattern', 'rank', 'contiguous')

    def __init__(self, arg_name, dtype, strides=None, rank=None, contiguous=None):
        assert isinstance(dtype, (ChoiceVar, str)), \
            f'@ati.tensor dtype must be a ChoiceVar or a literal type string, ' \
            f'got {dtype!r}'
        self.arg_name = arg_name
        self.dtype = dtype
        self.strides_pattern = strides
        self.rank = rank
        self.contiguous = contiguous          # int index | stride-name | None

    @property
    def is_literal_dtype(self) -> bool:
        return isinstance(self.dtype, str)

    @property
    def var_name(self) -> str:
        """The choice-variable name this tensor binds. A literal dtype is sugar
        for an anonymous single-choice variable named after the argument."""
        return self.dtype.name if isinstance(self.dtype, ChoiceVar) else self.arg_name

    def match_strides(self, param_names) -> list[str]:
        """Glob-match this tensor's stride parameters from the kernel signature,
        in signature order. Required — the whole-signature stride set cannot be
        partitioned by tensor without a per-tensor pattern."""
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

    def __repr__(self):
        return (f'TensorSpec({self.arg_name!r}, dtype={self.dtype!r}, '
                f'strides={self.strides_pattern!r}, rank={self.rank}, '
                f'contiguous={self.contiguous!r})')


class ScalarSpec:
    """One non-tensor argument binding (one `@ati.scalar`).

    Forms:
      scalar('Sm_scale', 'fp32')          -> type_='fp32'      (plain runtime)
      scalar('CAUSAL_TYPE', options=[..]) -> options=[..]      (enumerated)
      scalar('Max_seqlen_q', S)           -> dtype=ChoiceVar S (shared dimension)
      scalar('Num_head_q')                -> neither; type read from annotation
                                             at build time (Step 2.3)
    """

    __slots__ = ('arg_name', 'type_', 'dtype', 'options')

    def __init__(self, arg_name, type_or_var=None, options=None):
        self.arg_name = arg_name
        self.type_ = None
        self.dtype = None
        self.options = None
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
    def var_name(self) -> str:
        """The choice-variable name. A shared ChoiceVar uses its name; everything
        else is an anonymous variable named after the argument."""
        return self.dtype.name if self.dtype is not None else self.arg_name

    @property
    def has_explicit_type(self) -> bool:
        return self.type_ is not None or self.dtype is not None

    def __repr__(self):
        return (f'ScalarSpec({self.arg_name!r}, type_={self.type_!r}, '
                f'dtype={self.dtype!r}, options={self.options!r})')


def _is_numpy_array(x) -> bool:
    return type(x).__module__ == 'numpy' and type(x).__name__ == 'ndarray'


def tensor(arg_name, dtype, *, strides=None, rank=None, contiguous=None):
    """Bind a tensor argument to a dtype variable (or literal type string)."""
    return TensorSpec(arg_name, dtype, strides=strides, rank=rank,
                      contiguous=contiguous)


def scalar(arg_name, type_or_var=None, *, options=None):
    """Bind a non-tensor argument: a plain runtime scalar, an enumerated choice
    (options=), or a member of a shared choice variable."""
    return ScalarSpec(arg_name, type_or_var, options=options)


def overrides(targets, *, to, when):
    """Conditional override (agent-plans/ati_rev1.md §3.3): rewrite `targets` to
    `to` for functionals where `when` holds.

      ati.overrides('encoded_softmax', to=0,  when=ati.eq('RETURN_ENCODED_SOFTMAX', False))
      ati.overrides('Hdim_qk', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
      ati.overrides(['dropout_p', ...], to=0, when=ati.eq('ENABLE_DROPOUT', False))

    The shape of `to` selects the value kind:
      * str          -> VarRef (copy another variable's choice, e.g. BLOCK_DMODEL)
      * anything else -> literal (Choice.parse handles ints/bools/floats; `0` on a
                         tensor target is the constexpr-zero / former-CDETensor case)
    """
    value = VarRef(to) if isinstance(to, str) else to
    return Override(targets, when, value)
