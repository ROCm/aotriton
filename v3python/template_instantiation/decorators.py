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

from .ir import Override, VarRef, ValueFn
from .ir import eq, ne, lt, gt, le, ge   # predicate builders, re-exported as ati.*


class ChoiceVar:
    """A named choice variable shared by several arguments — the
    `template<typename T>` / TypeVar analogue. `tensor_dtype` and `choice_set`
    both produce one; `kind` records which authoring word created it (purely for
    diagnostics — Choice.parse handles tensor vs scalar literals uniformly).

    `signature_name` is the argument under which this variable is recorded in all
    PERSISTED forms — the compact signature string, which becomes the aks2 / zip
    entry name and the tuning-database row key. A multi-choice variable spanning
    several arguments MUST give it explicitly (it cannot be silently derived from
    spec order, or stored artifacts would shift); single-choice variables are
    trivial and exempt. `None` when not given — the builder resolves it."""

    __slots__ = ('name', 'choices', 'kind', 'signature_name')

    def __init__(self, name, choices, kind, signature_name=None):
        assert isinstance(name, str) and name, 'choice variable needs a name'
        self.name = name
        self.choices = list(choices)
        self.kind = kind          # 'dtype' | 'scalar'
        self.signature_name = signature_name

    def __call__(self, kernel):
        """Stacked-@ form: register this dtype/choice variable on the kernel below
        it (so `@ati.tensor_dtype('T_io', ...)` works as a decorator, and
        `@ati.tensor('Q', 'T_io', ...)` can refer to it by name). Mirrors
        TensorSpec/ScalarSpec.__call__."""
        from .describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return (f'ChoiceVar({self.name!r}, kind={self.kind}, '
                f'signature_name={self.signature_name!r}, choices={self.choices})')


def tensor_dtype(name, dtype, signature_name=None):
    """Declare a named tensor element-type variable. `dtype` is the choice set.
    `signature_name` is the argument that records this variable in persisted
    artifacts (compact signature, aks2/zip entry name, DB row key); required for
    multi-choice variables shared across several tensors."""
    return ChoiceVar(name, dtype, kind='dtype', signature_name=signature_name)


def choice_set(name, options, signature_name=None):
    """Declare a named scalar choice variable. `options` is the choice set.
    `signature_name` is the argument that records this variable in persisted
    artifacts (compact signature, aks2/zip entry name, DB row key)."""
    return ChoiceVar(name, options, kind='scalar', signature_name=signature_name)


class TensorSpec:
    """One tensor argument binding (one `@ati.tensor`).

    The first argument is either a single name or a LIST of names that share one
    dtype and shape but have NO strides — e.g. the rank-0 philox pointers
    (replaces grouping them under one frozenset). Strides require a per-tensor
    glob, so a name list forbids `strides=`/`contiguous=`.

    `dtype` is either a ChoiceVar (shared dimension) or a literal type string
    (anonymous single-choice variable). Strides are bound per tensor via a glob;
    rank is inferred from the matched stride count unless given explicitly."""

    __slots__ = ('arg_names', 'dtype', 'strides_pattern', 'rank', 'contiguous')

    def __init__(self, arg_name, dtype, strides=None, rank=None, contiguous=None):
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
            self.arg_names = tuple(arg_name)
        else:
            assert isinstance(arg_name, str), \
                f'@ati.tensor first arg must be a name or list of names, got {arg_name!r}'
            self.arg_names = (arg_name,)
        self.dtype = dtype
        self.strides_pattern = strides
        self.rank = rank
        self.contiguous = contiguous          # int index | stride-name | None

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

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this spec onto the kernel below it."""
        from .describe import accumulate_spec
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

    __slots__ = ('arg_names', 'type_', 'dtype', 'options')

    def __init__(self, arg_name, type_or_var=None, options=None):
        if isinstance(arg_name, (list, tuple)):
            assert arg_name, '@ati.scalar needs at least one argument name'
            assert all(isinstance(a, str) for a in arg_name), \
                '@ati.scalar name list must contain strings'
            self.arg_names = tuple(arg_name)
        else:
            assert isinstance(arg_name, str), \
                f'@ati.scalar first arg must be a name or list of names, got {arg_name!r}'
            self.arg_names = (arg_name,)
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
        from .describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return (f'ScalarSpec({self.arg_names!r}, type_={self.type_!r}, '
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


def derives(targets, *, to, when=None):
    """Derive `targets`' value from other functional state (agent-plans/ati_rev1.md
    §3.3). The single facade for both derive channels — the builder routes by
    target:
      * a kernel ARGUMENT target  -> applied in resolved[] (compiled signature),
        the former conditional/CC/CDETensor case (B, dropout_p, Hdim_qk, ...);
      * a PERF-SCHEMA target       -> applied in the perf layer (PERSISTENT_TYPE,
        NUM_XCDS), the former PROGRAMMATIC_PERFS / @ati.tune.derived case.

    `to` selects the value kind:
      * str          -> VarRef (copy another variable's choice, e.g. BLOCK_DMODEL)
      * callable     -> compute the value from the functional, `to(f)` — for a
                        value that is a function of functional state (e.g. NUM_XCDS
                        from arch, possibly several values 1/3/6/8). Fires
                        unconditionally unless `when` is also given.
      * anything else -> literal (Choice.parse handles ints/bools/floats; `0` on a
                         tensor target is the constexpr-zero / former-CDETensor case)

    `when` is an optional predicate gating the derive: a structured ati.eq/ne/...
    over a free choice axis, or a callable `f -> bool`. Omit it (callable `to`) to
    always fire.

      ati.derives('encoded_softmax', to=0, when=ati.eq('RETURN_ENCODED_SOFTMAX', False))
      ati.derives('Hdim_qk', to='BLOCK_DMODEL', when=ati.eq('PADDED_HEAD', False))
      ati.derives('PERSISTENT_TYPE', to=2, when=ati.ne('CAUSAL_TYPE', 0))
      ati.derives('NUM_XCDS', to=lambda f: {'gfx942': 8, 'gfx950': 8}.get(f.arch, 1))
    """
    if callable(to) and not isinstance(to, VarRef):
        value = ValueFn(to)
    elif isinstance(to, str):
        value = VarRef(to)
    else:
        value = to
    predicate = when if when is not None else _ALWAYS
    return Override(targets, predicate, value)


def _ALWAYS(functional):
    return True


# Back-compat alias; `derives` is the preferred facade.
overrides = derives


class DisableSpec:
    """One @ati.disable(when=callable): a predicate over the functional marking it
    excluded from generation (compiler/numerical correctness exclusion). Multiple
    compose with OR. The callable reads functional state (f.arch, f.choices.<var>).
    This is the user interface to is_functional_disabled."""

    __slots__ = ('when',)

    def __init__(self, when):
        assert callable(when), \
            f'@ati.disable(when=...) needs a callable f -> bool, got {when!r}'
        self.when = when

    def holds(self, functional) -> bool:
        return bool(self.when(functional))

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this spec onto the kernel below it."""
        from .describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return f'DisableSpec({getattr(self.when, "__name__", self.when)!r})'


def disable(when):
    """Disable the functionals where `when(functional)` is truthy — a
    compiler/numerical correctness exclusion that travels with the kernel
    description (the user interface to is_functional_disabled):

      ati.disable(when=lambda f: f.choices.CAUSAL_TYPE and f.choices.BIAS_TYPE != 0)
      ati.disable(when=lambda f: f.arch == 'gfx950' and f.choices.BLOCK_DMODEL == 16)
    """
    return DisableSpec(when)
