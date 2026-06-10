# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Smoke test for the ATI package skeleton (executive plan Step 0.2).

Asserts that every documented public name exists and that the unimplemented
stubs raise NotImplementedError. Run directly:

    python test/ati/test_skeleton.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import v3python.template_instantiation as ati

TOP_LEVEL_NAMES = [
    'tensor_dtype', 'choice_set', 'tensor', 'scalar',
    'overrides', 'eq', 'ne', 'lt', 'gt',
    'describe', 'operator', 'union_params', 'metro_kernel',
    'tune',
]

TUNE_NAMES = [
    'schema', 'configs', 'binning', 'fallback', 'derived', 'optune', 'Config',
]

BINNING_SELECTORS = ['le', 'gt', 'eq']


def _check_names():
    for name in TOP_LEVEL_NAMES:
        assert hasattr(ati, name), f'ati.{name} missing'
    for name in TUNE_NAMES:
        assert hasattr(ati.tune, name), f'ati.tune.{name} missing'
    for sel in BINNING_SELECTORS:
        assert hasattr(ati.tune.binning, sel), f'ati.tune.binning.{sel} missing'


def _check_stubs_raise():
    # Entry points not yet implemented must still fail loudly, not silently no-op.
    # (tensor_dtype/choice_set/tensor/scalar implemented in Step 2.1;
    #  overrides/eq/ne/lt/gt implemented in Step 2.2.)
    callables = [
        lambda: ati.describe(object()),
        lambda: ati.union_params([]),
        lambda: ati.tune.schema(object()),
        lambda: ati.tune.binning(key=ati.tune.binning.le),
        lambda: ati.tune.configs(lambda f: None),
    ]
    for call in callables:
        try:
            call()
        except NotImplementedError:
            continue
        raise AssertionError(f'expected NotImplementedError from {call!r}')


def main():
    _check_names()
    _check_stubs_raise()
    print('OK: ATI skeleton exposes all documented names; stubs raise.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
