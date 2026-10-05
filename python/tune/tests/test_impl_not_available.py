# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A forced impl_index that selects no kernel must stop the task.

Before, the tuning shim's hipErrorSharedObjectSymbolNotFound left NaN outputs,
compare() turned them into a null adiff, and compute_best_results rejected the
candidate as a broken kernel. On gfx1100 that silently discarded every
attn_fwd candidate past the padded twin's 64th.

pyaotriton needs a built library, so a stand-in exposing only hipError_t is
installed for the duration of each test.
"""

import enum
import sys
import types

import pytest

from aotriton.tune.kftdesc import ImplNotAvailable, KernelForTuneDescription
from aotriton.tune.tdesc import ImplSelector


class _FakeHipError(enum.IntEnum):
    hipSuccess = 0
    hipErrorInvalidImage = 200
    hipErrorSharedObjectSymbolNotFound = 302


@pytest.fixture
def fake_pyaotriton(monkeypatch):
    monkeypatch.setitem(sys.modules, 'pyaotriton', types.SimpleNamespace(hipError_t=_FakeHipError))


def _check(err, selector):
    # check_impl_available never touches `self`; avoid instantiating the ABC.
    KernelForTuneDescription.check_impl_available(None, err, selector, '02_irregular_hdim')


def test_out_of_range_impl_index_raises(fake_pyaotriton):
    with pytest.raises(ImplNotAvailable, match=r'attn_fwd=176 selected no kernel for test case 02_irregular_hdim'):
        _check(_FakeHipError.hipErrorSharedObjectSymbolNotFound, ImplSelector.parse_text('attn_fwd=176'))


def test_failed_compile_stays_a_rejectable_candidate(fake_pyaotriton):
    _check(_FakeHipError.hipErrorInvalidImage, ImplSelector.parse_text('attn_fwd=176'))


def test_success_passes(fake_pyaotriton):
    _check(_FakeHipError.hipSuccess, ImplSelector.parse_text('attn_fwd=0'))


def test_op_level_is_not_checked(fake_pyaotriton):
    # Op-level selection forces a backend, not an hsaco; its kernels use the DB.
    _check(_FakeHipError.hipErrorSharedObjectSymbolNotFound, ImplSelector.parse_text('op.attn_fwd=1'))
