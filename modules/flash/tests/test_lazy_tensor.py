# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""pyaotriton.lazy_tensor allocates dq_acc/delta through the libtorch stable ABI
(bindings/lazy_tensor/torch_stable_min.h), without torch headers. These tests
check that port against the libtorch that is actually loaded: the aten::empty
and aten::zero_ calls it makes must keep working on every supported torch.
"""

import ctypes

import pytest
import torch

from pyaotriton import T2, T4, DType, lazy_tensor

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_get_pointer.restype = ctypes.c_void_p
_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]


def _address(view):
    # TensorView.data_ptr is a void* in C++, which Python sees as a capsule.
    return _get_pointer(view.data_ptr, None) or 0


class _DeviceView:
    """Expose a pyaotriton TensorView to torch without copying."""

    def __init__(self, view):
        self.__cuda_array_interface__ = {
            'version': 2,
            'shape': tuple(view.sizes),
            'typestr': '<f4',
            'data': (_address(view), False),
            'strides': tuple(s * 4 for s in view.strides),
        }


def _as_torch(view):
    return torch.as_tensor(_DeviceView(view), device='cuda')


def _like(sizes):
    rank = len(sizes)
    klass = {2: T2, 4: T4}[rank]
    return klass(0, sizes, [0] * rank, DType.kFloat32)


def _dirty_allocator(sizes):
    # Leave nonzero data in a freed block, so an allocation that is not zeroed
    # shows up.
    torch.ones(*sizes, device='cuda')
    torch.cuda.synchronize()


def _row_major_strides(sizes):
    strides = [1] * len(sizes)
    for i in range(len(sizes) - 2, -1, -1):
        strides[i] = strides[i + 1] * sizes[i + 1]
    return tuple(strides)


@pytest.mark.parametrize('sizes', [(2, 3, 64, 16), (1, 1, 7, 1)])
def test_dq_acc_is_zeroed(sizes):
    _dirty_allocator(sizes)
    acc = lazy_tensor.dq_acc(_like(sizes), torch.cuda.current_device())
    view = acc.acquire()
    assert tuple(view.sizes) == sizes
    assert tuple(view.strides) == _row_major_strides(sizes)
    assert view.dtype == DType.kFloat32
    assert _address(view) != 0
    t = _as_torch(view)
    torch.cuda.synchronize()
    assert not t.any()
    acc.free()


@pytest.mark.parametrize('sizes', [(4, 128), (1, 5)])
def test_delta_is_usable(sizes):
    delta = lazy_tensor.delta(_like(sizes), torch.cuda.current_device())
    view = delta.acquire()
    assert tuple(view.sizes) == sizes
    assert tuple(view.strides) == _row_major_strides(sizes)
    t = _as_torch(view)
    t.fill_(3.0)
    torch.cuda.synchronize()
    assert bool((t == 3.0).all())
    delta.free()


def test_eager_is_passed_through():
    view = T2(0x1000, (4, 8), (8, 1), DType.kFloat32)
    eager = lazy_tensor.eager_delta(view)
    got = eager.acquire()
    assert _address(got) == 0x1000
    assert tuple(got.sizes) == (4, 8)


def test_free_releases_memory():
    sizes = (8, 4, 256, 64)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    for _ in range(50):
        acc = lazy_tensor.dq_acc(_like(sizes), torch.cuda.current_device())
        acc.acquire()
        acc.free()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs two GPUs')
def test_honors_device_index():
    sizes = (2, 2, 16, 16)
    for index in range(torch.cuda.device_count()):
        acc = lazy_tensor.dq_acc(_like(sizes), index)
        view = acc.acquire()
        with torch.cuda.device(index):
            t = _as_torch(view)
        assert t.device.index == index
        acc.free()
