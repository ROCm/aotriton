# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Smoke test for the ATI package skeleton (executive plan Step 0.2).

Asserts that every documented public name exists and that the unimplemented
stubs raise NotImplementedError. Run directly:

    python python/test/test_skeleton.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import aotriton.template_instantiation as ati

TOP_LEVEL_NAMES = [
    'type_var', 'scalar_var', 'tensor', 'scalar',
    'overrides', 'eq', 'ne', 'lt', 'gt',
    'disable', 'no_disable',
    'describe', 'start', 'operator', 'backend', 'union_params', 'metro_kernel',
    'tune',
]

TUNE_NAMES = [
    'schema', 'configs', 'binning', 'fallback', 'optune', 'Config',
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
    # (operator/backend are now implemented — Step 10 declarative operator surface.)
    callables = [
        lambda: ati.tune.optune(),
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
