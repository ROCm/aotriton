# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
ATI performance-tuning surface (ati.tune.*). Step 0.2 skeleton — see
agent-plans/ati_rev1.md §6 and agent-plans/ati_executive0.md.

    schema    - @dataclass perf schema -> perf params (§6)
    configs   - register gen_autotune_configs(f: Functional) (§6)
    binning   - @binning(key=algo) -> AUTOTUNE_KEYS; algorithm selectors are
                attributes: binning.le / binning.gt / binning.eq (§6)
    fallback  - @fallback(KEY=value) -> PARTIALLY_TUNED_FUNCTIONALS (§6)
    derived   - @derived(NAME=fn) -> PROGRAMMATIC_PERFS (§6)
    optune    - operator-level tuning keys (§4, §7 Q2)
    Config    - one perf point yielded by gen_autotune_configs (§6)
"""

__all__ = [
    'schema', 'configs', 'binning', 'fallback', 'derived', 'optune', 'Config',
]


def _stub(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f'ati.tune.{name} is not implemented yet (Step 0.2 skeleton). '
            f'See agent-plans/ati_executive0.md.')
    _raise.__name__ = name
    _raise.__qualname__ = f'tune.{name}'
    return _raise


class _BinningSelector:
    """A binning algorithm selector (ati.tune.binning.le/.gt/.eq). Carries only
    its identity until the tune layer maps it to the concrete Binning class."""
    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f'ati.tune.binning.{self.key}'


class _BinningDecorator:
    """`@ati.tune.binning(key=ati.tune.binning.le)` decorator, with the algorithm
    selectors exposed as attributes so both live under one name."""
    le = _BinningSelector('le')   # BinningLessOrEqual (default)
    gt = _BinningSelector('gt')   # BinningGreater (parity, not yet implemented)
    eq = _BinningSelector('eq')   # BinningExact

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            'ati.tune.binning is not implemented yet (Step 0.2 skeleton). '
            'See agent-plans/ati_executive0.md.')


schema = _stub('schema')
configs = _stub('configs')
binning = _BinningDecorator()
fallback = _stub('fallback')
derived = _stub('derived')
optune = _stub('optune')
Config = _stub('Config')
