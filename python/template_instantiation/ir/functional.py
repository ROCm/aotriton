# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Functional + the enumeration helpers.

A Functional is one fully-pinned instantiation of a kernel/operator, excluding perf:
every choice axis selected and all overrides applied, frozen into a resolved
{arg_name -> TypedChoice} table. `(arch_number, godel_number)` is its global identity
(arches share the godel space).

This is the SINGLE Functional class (the former low-level IR node and the codegen
adapter `AtiFunctional` are fused here): it carries the enumeration state AND the
codegen-facing signature/name/packing surface the generator reads. `meta_object` is
the owning Interface (kernel/operator) — None for bare IR-level use (tests). The
enumeration itself lives on `Interface.gen_functionals` (the classical shape); the
module-level `_resolve` / `_PredCtx` helpers are shared by it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .typed_choice import TypedChoice


class ChoiceVarAbsent(AttributeError):
    """Raised by ChoiceView when a predicate reads a choice variable the functional
    does not have. Subclasses AttributeError so getattr/hasattr duck-typing still
    behaves, but is_functional_disabled catches it specifically to emit a
    write-your-own-@ati.disable diagnostic (a cited disable predicate that reads a
    variable absent from the citing kernel)."""


class ChoiceView:
    """Ergonomic accessor over a Functional's pinned choices
    Step 1.5.md §6, ati+newbinds_rev1.md §5).

    Attribute access is keyed by *choice-variable name*: `f.choices.T_io` returns
    the variable's triton signature. `.tc(var)` returns the raw TypedChoice;
    `.arg(aname)` / `.arg_tc(aname)` read a resolved argument by real name."""

    __slots__ = ('_choice', '_resolved')

    def __init__(self, functional):
        self._choice = functional.choice       # var_name -> TypedChoice
        self._resolved = functional.resolved   # arg_name -> TypedChoice

    def tc(self, var):
        """The raw TypedChoice for a variable (for .itype / .type_enum / rank)."""
        if var not in self._choice:
            raise KeyError(
                f'{var!r} is not a choice variable; valid: {sorted(self._choice)}')
        return self._choice[var]

    def arg(self, aname):
        """Resolved (post-override) triton signature for an argument."""
        if aname not in self._resolved:
            raise KeyError(
                f'{aname!r} is not a resolved argument; '
                f'valid: {sorted(self._resolved)}')
        return self._resolved[aname].triton_compile_signature

    def arg_tc(self, aname):
        """Resolved (post-override) raw TypedChoice for an argument."""
        if aname not in self._resolved:
            raise KeyError(
                f'{aname!r} is not a resolved argument; '
                f'valid: {sorted(self._resolved)}')
        return self._resolved[aname]

    def __getattr__(self, var):
        choice = self._choice.get(var)
        if choice is None:
            raise ChoiceVarAbsent(
                f'{var!r} is not a choice variable of this functional; '
                f'valid: {sorted(self._choice)}')
        return choice.triton_compile_signature



class Functional:
    """One fully-pinned kernel/operator instantiation (excluding perf).

    State: `meta_object` (the owning Interface; None for bare IR tests), `arch`,
    `arch_number`, `godel_number`, `choice` ({var_name -> TypedChoice}), `resolved`
    ({arg_name -> TypedChoice}, post-override), `optimized_for` (the gpu list this
    functional is tuned for; may be empty). The rest is the codegen-facing surface
    (signatures, packing paths, per-arg docs)."""

    __slots__ = ('meta_object', 'arch', 'arch_number', 'godel_number',
                 'choice', 'resolved', '_optimized_for', '_choices_view')

    def __init__(self, *, meta_object, arch, arch_number, godel_number,
                 choice, resolved, optimized_for=()):
        self.meta_object = meta_object
        self.arch = arch
        self.arch_number = arch_number
        self.godel_number = godel_number
        self.choice = dict(choice)        # var_name -> TypedChoice (free + trivial)
        self.resolved = dict(resolved)    # arg_name -> TypedChoice (post-override)
        self._optimized_for = list(optimized_for)
        self._choices_view = None

    # --- pinned-choice accessors ---

    @property
    def choices(self) -> ChoiceView:
        """Cached ergonomic accessor; see ChoiceView."""
        if self._choices_view is None:
            self._choices_view = ChoiceView(self)
        return self._choices_view

    @property
    def identity(self):
        """Global identity: (arch_number, godel_number)."""
        return (self.arch_number, self.godel_number)

    # --- identity convenience (delegated to meta_object) ---

    @property
    def optimized_for(self):
        return self._optimized_for

    @property
    def noptimized_for(self):
        return len(self._optimized_for)

    @property
    def family(self):
        return self.meta_object.FAMILY

    @property
    def name(self):
        return self.meta_object.NAME

    @property
    def database_gpus(self):
        from aotriton.gpu_targets import AOTRITON_TUNING_DATABASE_FALLBACK
        return [
            (g, AOTRITON_TUNING_DATABASE_FALLBACK.get(g, g))
            for g in self._optimized_for
        ]

    # --- compact / fallback choices (multi-choice axes only) ---

    @property
    def compact_choices(self) -> dict:
        """label -> resolved TypedChoice for each multi-choice axis (the legacy
        compact_dict). The KEY is the axis's `signature_name` — the pure label used
        in persisted artifacts (aks2/zip entry, DB-row key), unrelated to wiring.
        The VALUE is looked up by the representative REAL argument (repr_arg);
        signature_name is never used to index the resolved table."""
        d = {}
        for ax in self.meta_object.axes_multi:
            d[ax.signature_name] = self.resolved[ax.repr_arg]
        return d

    @property
    def fallback_choices(self) -> dict:
        fb = self.meta_object.partially_tuned_functionals
        out = {}
        for k, tc in self.compact_choices.items():
            out[k] = fb.get(k, tc)
        return out

    # --- triton-compile-signature dicts (resolved per arg) ---

    def build_complete_tc_dict(self):
        """arg_name -> resolved TypedChoice for every argument (post-override)."""
        return dict(self.resolved)

    def pp_arg_doc(self, aname):
        """(is_constexpr, comment_value) for one launch argument's prepare_arguments
        entry, sourced from the resolved TypedChoice + the firing Override (no Bind):
          * non-constexpr   -> (False, None)
          * VarRef override -> (True, '<deferred var choices joined by />')
          * literal/plain   -> (True, '<baked value>')"""
        from .override import VarRef
        kdesc = self.meta_object
        # iter_launch_arguments emits apparel names; the IR tables are keyed on
        # real names. Map back before looking up.
        aname = kdesc.real_of(aname)
        choice = self.resolved[aname]
        if not choice.is_constexpr:
            return False, None
        ov = kdesc.override_for(aname)
        if ov is not None and isinstance(ov.value, VarRef):
            # Deferred to another variable: document its full choice list (the
            # legacy CDC behavior, e.g. Hdim_qk -> 16/32/.../512).
            src = kdesc.axis_by_var(ov.value.var_name)
            return True, '/'.join(
                str(c.triton_compile_signature) for c in src.choices)
        # Literal override or plain constexpr: the value actually baked in.
        return True, str(choice.triton_compile_signature)

    # --- core signatures (must match legacy bytes) ---

    @property
    def unified_signature(self) -> str:
        parts = []
        for ax in self.meta_object.axes_multi:
            # KEY = signature_name (pure persisted label); VALUE by repr_arg.
            tc = self.resolved[ax.repr_arg]
            parts.append(f'{ax.signature_name}={tc.testrun_entry_signature}')
        return ';'.join(parts)

    @property
    def human_readable_signature(self) -> str:
        # Best-effort: a C++ comment, not byte-load-bearing (per review). One line
        # per non-stride / non-hidden axis, representative arg + resolved value.
        lines = []
        for ax in self.meta_object.axes_all_ordered:
            if ax.is_stride and ax.kind == 'stride':
                # hidden u64 strides are omitted; baked (constexpr) strides shown
                if not self.resolved[ax.arg_names[0]].is_constexpr:
                    continue
            if ax.kind == 'stride_unit':
                continue
            lines.append(f'{ax.signature_name} = {self.resolved[ax.repr_arg]}')
        return 'Human-readable Signature \n// ' + '\n// '.join(lines)

    # --- file packing paths ---

    @property
    def filepack_ondisk_path(self):
        import hashlib
        from pathlib import Path
        from aotriton.gpu_targets import AOTRITON_ARCH_TO_DIRECTORY
        digest = hashlib.sha256(self.unified_signature.encode()).hexdigest()
        return (Path(AOTRITON_ARCH_TO_DIRECTORY[self.arch]) / self.meta_object.FAMILY
                / self.meta_object.NAME / digest)

    @property
    def filepack_inzip_name(self) -> str:
        return self.unified_signature

    @property
    def full_flatzip_path(self):
        from pathlib import Path
        from aotriton.gpu_targets import AOTRITON_ARCH_TO_DIRECTORY
        return (Path(AOTRITON_ARCH_TO_DIRECTORY[self.arch]) / self.meta_object.FAMILY
                / (self.meta_object.NAME + '.zip'))

    @property
    def tunecc_signature(self) -> str:
        return '#F;' + self.unified_signature + f';;arch={self.arch}'

    def __repr__(self):
        name = self.meta_object.NAME if self.meta_object is not None else '<bare>'
        return f'Functional({name!r}, arch={self.arch!r}, godel={self.godel_number})'


@dataclass(slots=True)
class _PredCtx:
    """Minimal functional-like context for evaluating override predicates during
    enumeration, before the Functional object exists. Exposes the `.choice` dict
    and `.arch` a callable predicate may read."""
    choice: dict[str, TypedChoice]      # var_name -> pinned TypedChoice
    arch: str


def _resolve(axes_all, overrides, picked, arch):
    """Step 3+4: fan out var->args (tensor ranks specialized per arg), then push
    overrides in declared order. Pure function of `picked` (+ arch dimension).
    Shared by Interface.gen_functionals (the enumeration lives there now)."""
    resolved = {}
    for axis in axes_all:
        nth = axis.choices.index(picked[axis.var_name])
        for arg in axis.arg_names:
            resolved[arg] = axis.choice_for_arg(nth, arg)
    ctx = _PredCtx(picked, arch)
    for ov in overrides:
        if ov.fires(ctx):
            c = ov.materialize(ctx)
            for t in ov.targets:
                resolved[t] = c
    return resolved
