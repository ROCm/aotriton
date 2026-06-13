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
        from .describe import accumulate_spec
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


def _is_callable_class_instance(when) -> bool:
    """True if `when` is an INSTANCE of a class defining __call__ (so it can extend
    a cited disable via super().__call__), as opposed to a bare function/lambda or a
    builtin. Functions, lambdas, and methods are not callable-class instances."""
    import types
    if isinstance(when, (types.FunctionType, types.LambdaType, types.MethodType,
                         types.BuiltinFunctionType)):
        return False
    # An instance whose own type defines __call__ (not a plain function object).
    return callable(when) and hasattr(type(when), '__call__') \
        and not isinstance(when, type)


class DisableSpec:
    """One @ati.disable(when=callable): a predicate over the functional marking it
    excluded from generation (compiler/numerical correctness exclusion). Multiple
    compose with OR. The callable reads functional state (f.arch, f.choices.<var>).
    This is the user interface to is_functional_disabled.

    `@ati.disable` is citeable (rev0 §4.5): a kernel with no local disable inherits
    the cited target's; a LOCAL disable REPLACES the cited one. To EXTEND (not
    replace) a cited disable, write it as a callable class, subclass it, and call
    super().__call__(f). `is_callable_class` records whether `when` structurally can
    do that; `override_ack` records the author's explicit affirmation that a bare
    callable intentionally overrides a cited disable (suppresses the §4.5 fatal
    error)."""

    __slots__ = ('when', 'is_callable_class', 'override_ack')

    def __init__(self, when, override_ack=False):
        assert callable(when), \
            f'@ati.disable(when=...) needs a callable f -> bool, got {when!r}'
        self.when = when
        self.is_callable_class = _is_callable_class_instance(when)
        self.override_ack = bool(override_ack)

    def holds(self, functional) -> bool:
        return bool(self.when(functional))

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this spec onto the kernel below it."""
        from .describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return f'DisableSpec({getattr(self.when, "__name__", self.when)!r})'


def disable(when, *, I_understand_this_overrides_cited_disable=False):
    """Disable the functionals where `when(functional)` is truthy — a
    compiler/numerical correctness exclusion that travels with the kernel
    description (the user interface to is_functional_disabled):

      ati.disable(when=lambda f: f.choices.CAUSAL_TYPE and f.choices.BIAS_TYPE != 0)
      ati.disable(when=lambda f: f.arch == 'gfx950' and f.choices.BLOCK_DMODEL == 16)

    When the kernel also has an @ati.cite, a LOCAL disable replaces the cited one.
    A bare lambda/function cannot call super() to extend the cited predicate, so it
    silently drops it — the builder raises a FATAL error unless you affirm the
    override with `I_understand_this_overrides_cited_disable=True`. A callable-class
    instance (which can extend via super().__call__) is accepted without the flag.
    (rev0 §4.5)
    """
    return DisableSpec(when,
                       override_ack=I_understand_this_overrides_cited_disable)


class CiteSpec:
    """One @ati.cite("<op>.<metro>"[.<kernel>]): a STRING reference to a metro (or
    one of its sub-kernels) whose instantiation practices the current kernel pulls
    in for any argument it shares by apparel name (agent-plans/
    ati_aux-kernel-xref_rev0.md §4.4). String-only to avoid circular imports between
    sibling kernel modules; resolved at build time against the `ops` registry."""

    __slots__ = ('target',)

    def __init__(self, target):
        assert isinstance(target, str) and target, (
            '@ati.cite needs a string "<op>.<metro>" or "<op>.<metro>.<kernel>" '
            f'(objects are disallowed to avoid circular imports), got {target!r}')
        parts = target.split('.')
        assert len(parts) in (2, 3), (
            f'@ati.cite target {target!r} must be "<op>.<metro>" or '
            f'"<op>.<metro>.<kernel>"')
        self.target = target

    @property
    def op_name(self):
        return self.target.split('.')[0]

    @property
    def metro_name(self):
        return self.target.split('.')[1]

    @property
    def kernel_name(self):
        """The cited sub-kernel name, or None for a whole-metro cite."""
        parts = self.target.split('.')
        return parts[2] if len(parts) == 3 else None

    def __call__(self, kernel):
        """Stacked-@ form: accumulate this cite onto the kernel below it."""
        from .describe import accumulate_spec
        return accumulate_spec(self, kernel)

    def __repr__(self):
        return f'CiteSpec({self.target!r})'


def cite(target):
    """Cite a metro (or one of its sub-kernels) to inherit its instantiation
    practices (rev0 §4.4):

      @ati.cite("op_attn_fwd.triton.attn_fwd")   # one sub-kernel
      @ati.cite("op_attn_bwd.triton_split")      # whole metro (merged interface)
    """
    return CiteSpec(target)


def source(path, name=None):
    """Innermost stacked-@ decorator: import the Triton source at `path` and return
    the kernel it defines, so the @ati.* decorators ABOVE stack onto the real kernel
    without copying it into the description (agent-plans/ati_modular_rev0.md §5a).

        @ati.kernel
        @ati.tensor('Q', 'T_io', strides='stride_q?', contiguous=-1)
        # ... more @ati.* specs ...
        @ati.source("../kernel/fwd_kernel.py")   # innermost, just above the def
        def attn_fwd():                          # placeholder: no args, body `pass`
            pass

    `path` is resolved relative to the DESCRIPTION file (the caller's __file__).
    The kernel symbol looked up in the imported module defaults to the placeholder
    `def`'s name; pass `name=` to override (the source filename, the kernel symbol,
    and the description module name are all independent). The returned object need
    not be a Triton kernel — any callable the builder can introspect.

    MUST be the innermost @ati.* (directly above the def): decorators apply
    bottom-up, so source() runs first and supplies the object every spec above then
    attaches to. A spec placed BELOW it would attach to the placeholder.
    """
    import inspect
    import importlib.util
    from pathlib import Path

    # Resolve `path` relative to the caller's file (the description module), not CWD.
    caller_file = inspect.stack()[1].filename
    base = Path(caller_file).resolve().parent
    src = (base / path).resolve()

    def _decorator(placeholder):
        sym = name or placeholder.__name__
        if not src.exists():
            raise AtiSourceError(
                f"@ati.source: kernel source {src} (from {path!r}, relative to "
                f"{base}) does not exist")
        # Put the kernel source dir on sys.path so its flat sibling imports
        # (`from fwd_kernel_inner import ...`) resolve, then import the kernel by its
        # stem. The kernel sources own the bare stem namespace (they import each
        # other by stem); the cached module is reused so all references share one
        # kernel object. NOTE: a DESCRIPTION module must NOT be imported under the
        # same bare stem as its kernel source — load descriptions under a namespace
        # (e.g. aot.<name>) so they never shadow sys.modules[<kernel stem>]; the
        # family loader does this.
        import sys
        import importlib.util
        srcdir = str(src.parent)
        if srcdir not in sys.path:
            sys.path.insert(0, srcdir)
        # Reuse a previously-loaded kernel module ONLY if its __file__ is exactly
        # this source path; otherwise load by EXPLICIT FILE PATH (not a sys.path
        # search) so the right file is loaded even when a same-stem description on
        # sys.path (aot/<stem>.py) would otherwise shadow the kernel (kernel/<stem>.py).
        cached = sys.modules.get(src.stem)
        if cached is not None and getattr(cached, '__file__', None) == str(src):
            mod = cached
        else:
            spec = importlib.util.spec_from_file_location(src.stem, src)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[src.stem] = mod      # so the kernel's sibling imports
                                             # (`from <stem> import ...`) and any
                                             # later @ati.source share this object
            spec.loader.exec_module(mod)
        if not hasattr(mod, sym):
            raise AtiSourceError(
                f"@ati.source({path!r}): module {src.name} has no symbol {sym!r} "
                f"(the placeholder def is named {placeholder.__name__!r}; pass "
                f"name= if the kernel symbol differs)")
        return getattr(mod, sym)

    return _decorator


class AtiSourceError(Exception):
    """A bad @ati.source: missing file or kernel symbol."""
