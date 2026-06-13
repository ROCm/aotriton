# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Binning selectors and the @ati.tune.binning decorator (agent-plans/ati_rev1.md
§6). The algorithm selectors are attributes on the `binning` callable so both the
decorator and its selectors live under one name:

    @ati.tune.binning(Max_seqlen_q=ati.tune.binning.le,
                      Max_seqlen_k=ati.tune.binning.le)

Selector -> concrete aotriton.autotune.Binning class is resolved in Step 3.2.
"""


class BinningSelector:
    """A binning-algorithm selector (binning.le / .gt / .eq). Carries only its
    identity until the tune layer maps it to a concrete Binning class."""
    __slots__ = ('key',)

    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f'ati.tune.binning.{self.key}'


class Binning:
    """The @ati.tune.binning(key=selector) decorator, with the algorithm
    selectors exposed as attributes so both live under one name. Step 3.2 wires
    the call to attach the binning map to the kernel."""
    le = BinningSelector('le')   # BinningLessOrEqual (default)
    gt = BinningSelector('gt')   # BinningGreater (parity, not yet implemented)
    eq = BinningSelector('eq')   # BinningExact

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            'ati.tune.binning is not implemented yet (Step 3.2). '
            'See agent-plans/ati_executive0.md.')


binning = Binning()
