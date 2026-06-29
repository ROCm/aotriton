# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
The `ati.tune` namespace facade.

`tune` was a package whose only job, after the pipeline-stage reorg, was to re-export
the tuning surface. The actual code lives in its pipeline stages now:
- decorators/tune.py: the author-facing factories (schema/configs/binning/fallback).
- specs/tune.py: the passive spec records (TuneSpec/PerfSchema/Config/
  ConfigsSpec/BinningSpec/FallbackSpec/BinningSelector).

So `ati.tune` is just a grouping label, not a layer — a plain namespace object backed
by those two modules. `ati.tune.schema(...)`, `ati.tune.binning.le`, `ati.tune.Config`,
`hasattr(ati.tune, 'optune')` all work as before. Internal code imports the records
from `specs.tune` directly; only the author-facing `ati.tune.*` surface goes here.
"""

from types import SimpleNamespace

from .decorators.tune import schema, configs, binning, fallback
from .specs.tune import (
    PerfSchema, BinningSelector,
    Config, TuneSpec, ConfigsSpec, BinningSpec, FallbackSpec,
)


def _stub(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f'ati.tune.{name} is not implemented yet. '
            f'See agent-plans/ati_executive0.md.')
    _raise.__name__ = name
    _raise.__qualname__ = f'tune.{name}'
    return _raise


tune = SimpleNamespace(
    schema=schema, configs=configs, binning=binning, fallback=fallback,
    PerfSchema=PerfSchema, BinningSelector=BinningSelector,
    Config=Config, TuneSpec=TuneSpec,
    ConfigsSpec=ConfigsSpec, BinningSpec=BinningSpec, FallbackSpec=FallbackSpec,
    # operator-level tuning keys; implemented with the operator in Phase 5.
    optune=_stub('optune'),
)
