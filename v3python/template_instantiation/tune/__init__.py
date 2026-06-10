# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI performance-tuning layer (ati.tune.*).

The perf schema (Step 3.1) and the configs/binning/fallback/derived registration
(Step 3.2) live here. This package supersedes the Step 0.2 stub module
(template_instantiation/tune.py was a flat stub); the public surface is unchanged.
"""

from .schema import PerfParam, PerfSchema, schema
from .binning import binning, Binning, BinningSelector

__all__ = [
    'PerfParam', 'PerfSchema', 'schema',
    'binning', 'Binning', 'BinningSelector',
    'configs', 'fallback', 'derived', 'optune', 'Config',
]


def _stub(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f'ati.tune.{name} is not implemented yet. '
            f'See agent-plans/ati_executive0.md.')
    _raise.__name__ = name
    _raise.__qualname__ = f'tune.{name}'
    return _raise


# Implemented in Step 3.2.
configs = _stub('configs')
fallback = _stub('fallback')
derived = _stub('derived')
optune = _stub('optune')
Config = _stub('Config')
