# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tune decorator factories (pipeline Stage 1; agent-plans/ati_rev1.md §6).

The author-facing entry points of ati.tune.*: schema / configs / binning /
fallback. Each builds a passive spec record (specs/tune.py) which doubles as a
stacked-@ decorator. The binning algorithm selectors (.le/.gt/.eq) hang off the
`binning` callable so the decorator and its selectors live under one name:

    @ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                      Max_seqlen_k=ati.tune.binning.le)
"""

from ..specs.tune import (
    BinningSelector, ConfigsSpec, BinningSpec, FallbackSpec, build_schema,
)


def schema(dataclass):
    """ati.tune.schema(PerfDataclass): build the PerfSchema from a @dataclass whose
    fields (annotated with numpy dtypes / bool) are the kernel's perf parameters."""
    return build_schema(dataclass)


def configs(generator):
    """ati.tune.configs(gen_autotune_configs): register the per-functional perf
    config generator."""
    return ConfigsSpec(generator)


class _BinningDecorator:
    """ati.tune.binning(key=selector) builds a BinningSpec, while
    ati.tune.binning.le/.gt/.eq are the algorithm selectors."""
    le = BinningSelector('le')   # BinningLessOrEqual (default)
    gt = BinningSelector('gt')   # BinningGreater (parity, not yet implemented)
    eq = BinningSelector('eq')   # BinningExact

    def __call__(self, **keys):
        return BinningSpec(keys)


binning = _BinningDecorator()


def fallback(**values):
    """ati.tune.fallback(KEY=value, ...) -> PARTIALLY_TUNED_FUNCTIONALS."""
    return FallbackSpec(values)
