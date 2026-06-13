# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Choice — a single concrete instantiation value (ATI executive plan Step 1.1).

A Choice is one settled value an argument can take: a tensor dtype (`*fp16:16`),
a constexpr (`0`, `False`), or a scalar type (`i32`). It is the unit an Axis
enumerates over (see agent-plans/ati+newbinds_rev1.md §2.1).

Rather than reinvent type strings, Choice is a thin wrapper over the existing
`ir.typed_choice` (TC) settled objects: it adapts what enumeration and
codegen need (triton signature, C itype, dtype enum) and adds rank handling,
since in the new IR a tensor's rank lives on the Axis, not the dtype.

NOTE: the new IR deliberately does NOT use the TC `resolve`/`resolve_rank`/
`tc_dict`/`_specialized` machinery. Those served the old conditional-value and
per-name-RANKS design, both of which are replaced (overrides for conditionals,
Axis-owned ranks for shape). `with_rank()` builds a fresh ranked tensor TC
directly instead. Do not reach for `resolve*` when extending this layer.

WHY a wrapper instead of extending TypedChoice (and its lifecycle):
  Choice owns only the semantics the new IR needs but the old IR never had:
    * value equality + hashing (TypedChoice uses identity equality; the old path
      compares triton_compile_signature strings / np.unique on dataframes, never
      the TC objects). enumerate_functionals, the godel bijection, and the
      f.choices view all key on Choice value equality.
    * a narrow rank surface (with_rank) instead of the pull-model
      resolve_rank/_specialized API.
  Adding those to TypedChoice means changing a class ~30 shipping subclasses
  still depend on, mid-migration, while the golden gate requires the old path to
  stay byte-for-byte identical. So Choice is a deliberately TEMPORARY adapter.
  Plan: keep Choice until the migration is complete; then merge its features into
  TypedChoice and delete this wrapper. See executive plan Step 6.3.
"""

from . import typed_choice as TC


class Choice:
    """Wraps a settled TypedChoice. For tensors the rank may be unbound until the
    owning Axis specializes it via with_rank()."""

    __slots__ = ('_tc',)

    def __init__(self, tc: TC.TypedChoice):
        assert isinstance(tc, TC.TypedChoice), \
            f'Choice wraps a settled TypedChoice, got {tc!r}'
        assert not isinstance(tc, TC.ConditionalChoice), \
            'Choice cannot wrap a ConditionalChoice; the new IR has no deferred ' \
            'choices (overrides replace them).'
        self._tc = tc

    # ---- construction ----

    @classmethod
    def parse(cls, spec):
        """Build a Choice from an authoring literal: a type string (`'*fp16:16'`,
        `'i32'`), a python scalar (`0`, `False`), or an already-built TypedChoice.
        Mirrors the existing parse_choices single-element path."""
        if isinstance(spec, TC.TypedChoice):
            return cls(spec)
        if isinstance(spec, str):
            return cls(TC.parse_complex(spec))
        # python scalars (int/bool/float/np scalar) -> constexpr via the guessers
        tcs = TC.parse_choices([spec])
        assert len(tcs) == 1
        return cls(tcs[0])

    # ---- underlying object ----

    @property
    def tc(self) -> TC.TypedChoice:
        return self._tc

    # ---- type/shape facts ----

    @property
    def is_tensor(self) -> bool:
        return self._tc.is_tensor

    @property
    def is_constexpr(self) -> bool:
        return isinstance(self._tc, TC.constexpr_base)

    def with_rank(self, rank: int) -> 'Choice':
        """Return a Choice whose tensor TC is specialized to a concrete rank.
        Non-tensor choices ignore rank and return self."""
        if not self._tc.is_tensor:
            return self
        # Preserve subclass (tensor vs lazy_tensor) and element type.
        ranked = self._tc.__class__(elem_ty=self._tc._elem_ty, rank=rank)
        return Choice(ranked)

    # ---- codegen-facing strings ----

    @property
    def triton_compile_signature(self):
        """What guides Triton compilation: `*fp16:16`, `i32`, or a constexpr value."""
        return self._tc.triton_compile_signature

    @property
    def itype(self) -> str:
        """The C type for a params-struct field / argument."""
        return self._tc.itype

    @property
    def type_enum(self) -> str:
        """DType::kFloat16 etc. Only meaningful for typed (non-constexpr) args."""
        return self._tc.type_enum

    @property
    def infotext(self) -> str:
        """Literal used inside generated C choice tables."""
        return self._tc.infotext

    # ---- identity ----

    def __eq__(self, other):
        if not isinstance(other, Choice):
            return NotImplemented
        return (self._tc.__class__ == other._tc.__class__ and
                self.triton_compile_signature == other.triton_compile_signature)

    def __hash__(self):
        return hash((self._tc.__class__, str(self.triton_compile_signature)))

    def __repr__(self):
        return f'Choice({self.triton_compile_signature!r})'
