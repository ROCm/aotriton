# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Passive tuning spec records (pipeline Stage 2; agent-plans/ati_rev1.md §6).

The data side of ati.tune.*: the perf schema (PerfSchema/PerfParam), the stacked-@
spec records (ConfigsSpec/BinningSpec/FallbackSpec), the binning selector value
object (BinningSelector), the per-functional Config a generator yields, and the
kernel's collected TuneSpec. The decorator FACTORIES that produce these
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


class PerfParam:
    """One performance parameter: its name, the constexpr TypedChoice class fixing
    its C struct width, and an optional default value (from the @dataclass field
    default). The default is the value used by the *empty/untuned* path (e.g.
    bwd_preprocess, or a functional with no DB row); tuned kernels read their
    value from the DB instead, so the default is rarely exercised.

    Not a @dataclass: the `_NO_DEFAULT` sentinel default value would be exposed as a
    class attribute clashing with the dataclass `default` field name."""
    __slots__ = ('name', 'tcc', 'default')

    _NO_DEFAULT = object()

    def __init__(self, name, tcc, default=_NO_DEFAULT):
        self.name = name
        self.tcc = tcc          # a TCC.*_t class
        self.default = default  # python value | _NO_DEFAULT

    @property
    def has_default(self) -> bool:
        return self.default is not PerfParam._NO_DEFAULT

    def default_choice(self):
        """The settled constexpr TypedChoice for the default value."""
        assert self.has_default, \
            f'perf param {self.name!r} has no default; give the @dataclass field ' \
            f'a default value for the untuned/empty path'
        return self.tcc(self.default)

    def choice_for(self, value):
        """A settled constexpr TypedChoice of this param's declared width."""
        return self.tcc(value)

    @property
    def itype(self) -> str:
        # Width-only instance to read the C itype (value is irrelevant).
        return self.tcc(0).itype

    def __repr__(self):
        return f'PerfParam({self.name!r}, {self.tcc.__name__}, default={self.default!r})'


class PerfSchema(StackedSpec):
    """The ordered perf parameters of a kernel, derived from a @dataclass."""
    __slots__ = ('dataclass', 'params')

    def __init__(self, dataclass, params):
        self.dataclass = dataclass
        self.params = params            # list[PerfParam], field order

    @property
    def names(self):
        return [p.name for p in self.params]

    def __repr__(self):
        return (f'PerfSchema({self.dataclass.__name__!r}, '
                f'{[p.name for p in self.params]})')


def build_schema(dataclass) -> PerfSchema:
    assert dataclasses.is_dataclass(dataclass), \
        f'ati.tune.schema expects a @dataclass, got {dataclass!r}'
    params = []
    for f in dataclasses.fields(dataclass):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
            default = f.default_factory()
        else:
            default = PerfParam._NO_DEFAULT
        params.append(PerfParam(f.name, _resolve_tcc(f.name, f.type), default))
    return PerfSchema(dataclass, params)


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
        self.schema = None        # PerfSchema
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
