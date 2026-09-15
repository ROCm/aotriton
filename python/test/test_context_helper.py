# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""`ati.context_helper()` is a scalar-only `wires_to=` value.

A helper's result lands in the context's `scratch_params` and is CAST into the
kernarg slot. There is no tensor form of that: a wired tensor is dereferenced
(`params.X->kparam_data_ptr()`), so a tensor helper has nothing to store and
nothing to dereference. Rejecting it at the decorator keeps the failure at the
description, where the author can read it -- the generator would otherwise
ignore the wiring and silently emit `params.<arg>` for an operand the operator
does not have.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

import aotriton.template_instantiation as ati
from aotriton.template_instantiation.ir.context_helper import ContextHelper


def test_scalar_takes_a_context_helper():
    s = ati.scalar('num_seqlens', 'i32',
                   wires_to=ati.context_helper('flyc_num_seqlens'))
    assert s.wires_to == ContextHelper('flyc_num_seqlens')


def test_tensor_refuses_a_context_helper():
    with pytest.raises(AssertionError, match='scalar-only'):
        ati.tensor('Q', '*fp16:16', wires_to=ati.context_helper('flyc_q'))


def test_tensor_still_takes_an_operand_rename():
    # The guard is on the helper, not on wires_to: a rename is the common case.
    assert ati.tensor('Q', '*fp16:16', wires_to='Q_operand').wires_to == 'Q_operand'


def main():
    """Standalone runner; `pytest.raises` is a plain context manager."""
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failures = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f'FAIL: {fn.__name__}: {type(e).__name__}: {e}')
    if failures:
        print(f'FAILED: {failures} of {len(fns)} context_helper tests.')
        return 1
    print(f'OK: {len(fns)} context_helper tests passed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
