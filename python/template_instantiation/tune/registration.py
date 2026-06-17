# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tune registration decorators (executive plan Step 3.2; agent-plans/ati_rev1.md
§6): configs / binning / fallback / derived, plus the Config a generator yields.

Each decorator is a callable spec-record (the same pattern as @ati.tensor): when
written as a stacked `@`, calling it on the kernel accumulates it; when collected
by describe(), it is partitioned into the kernel's TuneSpec. Mapping a binning
selector to a concrete aotriton.autotune Binning class, and wiring the TuneSpec
into the codegen adapter, is Phase 4 — here we record and expose the metadata.
"""

from .binning import BinningSelector, binning as _binning_singleton

# Compiler options carried alongside perf fields in a Config (not perf struct
# fields). Mirrors aotriton.kernel.ksignature.COMPILER_OPTIONS.
_COMPILER_OPTIONS = ('waves_per_eu', 'num_warps', 'num_stages')
_CONFIG_RESERVED = _COMPILER_OPTIONS + ('num_ctas', 'maxnreg', 'pre_hook')


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

    __slots__ = ('kwargs', 'num_warps', 'num_stages', 'num_ctas',
                 'maxnreg', 'pre_hook')

    def __init__(self, kwargs, num_warps=4, num_stages=2, num_ctas=1,
                 maxnreg=None, pre_hook=None):
        self.kwargs = dict(kwargs)
        self.num_warps = num_warps
        self.num_stages = num_stages
        self.num_ctas = num_ctas
        self.maxnreg = maxnreg
        self.pre_hook = pre_hook

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


class _TuneRecord:
    """Base for the stacked-@ tune spec records: callable to accumulate."""
    def __call__(self, kernel):
        from ..describe import accumulate_spec
        return accumulate_spec(self, kernel)


class ConfigsSpec(_TuneRecord):
    """@ati.tune.configs(gen_autotune_configs): registers the per-functional perf
    config generator. The generator takes a Functional and yields Config."""
    __slots__ = ('generator',)

    def __init__(self, generator):
        assert callable(generator), \
            f'ati.tune.configs expects a callable, got {generator!r}'
        self.generator = generator

    def __repr__(self):
        return f'ConfigsSpec({getattr(self.generator, "__name__", self.generator)!r})'


class BinningSpec(_TuneRecord):
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


class FallbackSpec(_TuneRecord):
    """@ati.tune.fallback(KEY=value, ...) -> PARTIALLY_TUNED_FUNCTIONALS."""
    __slots__ = ('values',)

    def __init__(self, values):
        self.values = dict(values)

    def __repr__(self):
        return f'FallbackSpec({self.values})'


# --- public decorator entry points ---

def configs(generator):
    return ConfigsSpec(generator)


class _BinningDecorator:
    """Replaces the placeholder Binning.__call__: ati.tune.binning(key=selector)
    builds a BinningSpec, while ati.tune.binning.le/.gt/.eq stay as selectors."""
    le = _binning_singleton.le
    gt = _binning_singleton.gt
    eq = _binning_singleton.eq

    def __call__(self, **keys):
        return BinningSpec(keys)


binning = _BinningDecorator()


def fallback(**values):
    return FallbackSpec(values)


# --- the kernel's collected tuning metadata ---

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
