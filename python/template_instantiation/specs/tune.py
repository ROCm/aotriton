# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Passive tuning spec records (pipeline Stage 2; agent-plans/ati_rev1.md §6).

The data side of ati.tune.*: the perf schema (a SYNTHESIZED dataclass + PerfSchema
envelope), the stacked-@ spec records (ConfigsSpec/BinningSpec/FallbackSpec), the
binning selector value object (BinningSelector), the per-functional Config a generator
yields, and the kernel's collected TuneSpec. The decorator FACTORIES that produce these
(ati.tune.schema/configs/binning/fallback) live in decorators/tune.py.

Each spec record is callable so it doubles as a stacked-@ decorator (the same
pattern as @ati.tensor): calling it on a kernel accumulates it onto the pending
list; describe()/the finalizer partitions it into the kernel's TuneSpec.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from ..ir.typed_choice import constexpr as TCC
from ..ir.typed_choice import NUMPY_TO_CONSTEXPR
from .base import StackedSpec


# --- binning selector value object -----------------------------------------

class BinningSelector:
    """A binning-algorithm selector (binning.le / .gt / .eq). Carries only its
    identity until the tune layer maps it to a concrete Binning class."""
    __slots__ = ('key',)

    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f'ati.tune.binning.{self.key}'


# --- perf schema -----------------------------------------------------------


def _resolve_tcc(field_name, annotation):
    """Map a dataclass field annotation to its constexpr TypedChoice class."""
    if annotation is bool or annotation is np.bool_:
        return TCC.bool_t
    # numpy scalar types may be given as the type (np.int16) or dtype(np.int16).
    np_type = None
    if isinstance(annotation, type) and issubclass(annotation, np.generic):
        np_type = annotation
    else:
        try:
            np_type = np.dtype(annotation).type
        except TypeError:
            np_type = None
    if np_type in NUMPY_TO_CONSTEXPR:
        return NUMPY_TO_CONSTEXPR[np_type]
    raise TypeError(
        f"perf schema field {field_name!r}: annotation {annotation!r} is not a "
        f"supported perf dtype; use a numpy integer dtype (np.int8/int16/...) or "
        f"bool")


class PerfStructBase:
    """Mixin for the SYNTHESIZED perf struct (built by build_schema). The struct's
    fields ARE the perf params: each field is annotated with its constexpr TypedChoice
    class (the C width) and carries the python default (the untuned/empty-path value).

    The class doubles as the schema (the classmethods below query the fields); an
    INSTANCE is one perf bind row (every field set to a settled TypedChoice), iterated
    via items(). This single object replaces the old PerfParam + _PerfParamShim +
    _PerfBind wrapper stack."""

    @classmethod
    def param_names(cls) -> list[str]:
        return [f.name for f in dataclasses.fields(cls)]

    @classmethod
    def tcc_of(cls, name):
        """The constexpr TypedChoice CLASS for a field (its annotation). We annotate
        synthesized fields with the class object itself, so field.type IS the class."""
        tcc = cls.__dataclass_fields__[name].type
        assert isinstance(tcc, type), (
            f'perf field {name!r} annotation is {tcc!r}, expected a constexpr class '
            f'(stringized annotations are not used on synthesized perf structs)')
        return tcc

    @classmethod
    def itype_of(cls, name) -> str:
        # Width-only instance to read the C itype (value is irrelevant).
        return cls.tcc_of(name)(0).itype

    @classmethod
    def default_value(cls, name):
        return cls.__dataclass_fields__[name].default

    @classmethod
    def choice_for(cls, name, value):
        """A settled constexpr TypedChoice of this field's declared width."""
        return cls.tcc_of(name)(value)

    @classmethod
    def default_choice(cls, name):
        """The settled constexpr TypedChoice for the field's default value."""
        return cls.choice_for(name, cls.default_value(name))

    @classmethod
    def cfields(cls, index_of=None):
        """One cfield per perf field (C struct layout). `index_of` is an optional
        name -> int callback for the kernel's argument index (default -1). The caller
        owns any ordering (e.g. the nbits-descending sort in codegen)."""
        from ..ir.cfield import cfield
        out = []
        for name in cls.param_names():
            tc = cls.tcc_of(name)(0)
            out.append(cfield(ctype=tc.itype, aname=name,
                              index=(index_of(name) if index_of else -1),
                              nbits=tc.NBITS or 0))
        return out

    def items(self):
        """(name, settled TypedChoice) for each field of this bind row."""
        for f in dataclasses.fields(self):
            yield f.name, getattr(self, f.name)


class PerfSchema(StackedSpec):
    """Stage-2 spec record for @ati.tune.schema: a thin envelope around the synthesized
    perf struct CLASS (PerfStructBase subclass). Kept a StackedSpec so the stacked-@
    `__call__` accumulate works; the finalizer stores `.struct` on TuneSpec.schema."""
    __slots__ = ('struct',)

    def __init__(self, struct):
        self.struct = struct

    def __repr__(self):
        return (f'PerfSchema({self.struct.__name__!r}, '
                f'{self.struct.param_names()})')


def build_schema(src) -> PerfSchema:
    """Synthesize a perf struct from the author's @dataclass: replace each numpy field
    type with its constexpr TypedChoice class, keeping the field's default. Every field
    MUST declare a default (the untuned/empty-path value)."""
    assert dataclasses.is_dataclass(src), \
        f'ati.tune.schema expects a @dataclass, got {src!r}'
    specs = []
    for f in dataclasses.fields(src):
        tcc = _resolve_tcc(f.name, f.type)
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            default = f.default_factory()
        else:
            raise TypeError(
                f"perf schema {src.__name__!r}: field {f.name!r} needs a default "
                f"(the untuned/empty-path value); give it `= <value>` in the dataclass")
        specs.append((f.name, tcc, default))     # annotation = the constexpr CLASS
    struct = dataclasses.make_dataclass(src.__name__, specs,
                                        bases=(PerfStructBase,), slots=True)
    return PerfSchema(struct)


# A canonical empty perf struct for kernels with no @ati.tune.schema: its param_names()
# / items() are empty, so the perf code paths need no None-guards.
EMPTY_PERF_STRUCT = dataclasses.make_dataclass('NoPerf', [], bases=(PerfStructBase,),
                                               slots=True)


# --- per-functional Config -------------------------------------------------

# Compiler options carried alongside perf fields in a Config (not perf struct
# fields). Mirrors aotriton.kernel.ksignature.COMPILER_OPTIONS.
_COMPILER_OPTIONS = ('waves_per_eu', 'num_warps', 'num_stages')
_CONFIG_RESERVED = _COMPILER_OPTIONS + ('num_ctas', 'maxnreg', 'pre_hook')


@dataclass(slots=True)
class Config:
    """One perf point yielded by a gen_autotune_configs generator.

    The constructor mirrors triton.Config / the existing aotriton.autotune.Config
    to minimize API churn in the per-kernel generators (which already write
    `yield Config(kw, num_stages=4, num_warps=8)`):

        yield ati.tune.Config(kw, num_warps=warps, num_stages=stages)

    where `kw` is the perf-field dict (and, as in triton.Config for AMD,
    `waves_per_eu` is carried INSIDE kw).

    Config then exposes the AOTriton classification downstream:
      * `.perf`  - constexpr KERNEL ARGUMENTS (kw minus waves_per_eu), baked into
                   the kernel signature.
      * `.copts` - COMPILER OPTIONS (waves_per_eu, num_warps, num_stages), passed
                   to the Triton compiler, not the kernel.

    NOTE on waves_per_eu: triton.Config stuffs it into kwargs as if it were a
    kernel argument; AOTriton has always treated it as a compiler option (see
    COMPILER_OPTIONS in aotriton.kernel.ksignature). So we accept it inside kw for
    call-site compatibility but route it into .copts, never .perf."""

    kwargs: dict[str, object]               # perf-field dict (waves_per_eu inside)
    num_warps: int = 4
    num_stages: int = 2
    num_ctas: int = 1
    maxnreg: int | None = None
    pre_hook: object = None

    def __post_init__(self):
        self.kwargs = dict(self.kwargs)     # defensive copy of the perf-field dict

    @property
    def waves_per_eu(self):
        return self.kwargs.get('waves_per_eu', 0)

    @property
    def perf(self) -> dict:
        """Constexpr kernel-argument fields (kwargs minus the compiler option
        waves_per_eu)."""
        return {k: v for k, v in self.kwargs.items() if k != 'waves_per_eu'}

    @property
    def copts(self) -> dict:
        return {'waves_per_eu': self.waves_per_eu,
                'num_warps': self.num_warps,
                'num_stages': self.num_stages}

    def __repr__(self):
        return (f'Config({self.kwargs}, num_warps={self.num_warps}, '
                f'num_stages={self.num_stages})')


# --- stacked-@ tune spec records -------------------------------------------

class ConfigsSpec(StackedSpec):
    """@ati.tune.configs(gen_autotune_configs): registers the per-functional perf
    config generator. The generator takes a Functional and yields Config."""
    __slots__ = ('generator',)

    def __init__(self, generator):
        assert callable(generator), \
            f'ati.tune.configs expects a callable, got {generator!r}'
        self.generator = generator

    def __repr__(self):
        return f'ConfigsSpec({getattr(self.generator, "__name__", self.generator)!r})'


class BinningSpec(StackedSpec):
    """@ati.tune.binning(key=selector, ...): autotune binning keys -> selectors."""
    __slots__ = ('keys',)

    def __init__(self, keys):
        for k, sel in keys.items():
            assert isinstance(sel, BinningSelector), (
                f'ati.tune.binning({k}=...) expects a selector '
                f'(ati.tune.binning.le/.gt/.eq), got {sel!r}')
        self.keys = dict(keys)

    def __repr__(self):
        return f'BinningSpec({self.keys})'


class FallbackSpec(StackedSpec):
    """@ati.tune.fallback(KEY=value, ...) -> PARTIALLY_TUNED_FUNCTIONALS."""
    __slots__ = ('values',)

    def __init__(self, values):
        self.values = dict(values)

    def __repr__(self):
        return f'FallbackSpec({self.values})'


# --- the kernel's collected tuning metadata --------------------------------

class TuneSpec:
    """All tuning metadata gathered for one kernel (lives on KernelSpec.tune)."""
    __slots__ = ('schema', 'configs', 'binning', 'fallback')

    def __init__(self):
        self.schema = None        # the synthesized perf struct CLASS (PerfStructBase)
        self.configs = None       # generator callable
        self.binning = {}         # key -> BinningSelector
        self.fallback = {}        # key -> value

    @property
    def is_tunable(self) -> bool:
        return self.configs is not None

    def __repr__(self):
        return (f'TuneSpec(schema={self.schema is not None}, '
                f'tunable={self.is_tunable}, binning={list(self.binning)}, '
                f'fallback={list(self.fallback)})')
