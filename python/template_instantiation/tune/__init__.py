# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI performance-tuning layer (ati.tune.*).

The perf schema and the configs/binning/fallback registration live here. Per-functional
derived perf values use @ati.derives (the unified derive facade), not a separate
tune.derived channel.
"""

from .schema import PerfParam, PerfSchema, schema
from .binning import BinningSelector
from .registration import (
    Config, TuneSpec,
    ConfigsSpec, BinningSpec, FallbackSpec,
    configs, binning, fallback,
)

__all__ = [
    'PerfParam', 'PerfSchema', 'schema',
    'binning', 'BinningSelector',
    'configs', 'fallback', 'optune', 'Config',
    'TuneSpec', 'ConfigsSpec', 'BinningSpec', 'FallbackSpec',
]


def _stub(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f'ati.tune.{name} is not implemented yet. '
            f'See agent-plans/ati_executive0.md.')
    _raise.__name__ = name
    _raise.__qualname__ = f'tune.{name}'
    return _raise


# operator-level tuning keys; implemented with the operator in Phase 5.
optune = _stub('optune')
