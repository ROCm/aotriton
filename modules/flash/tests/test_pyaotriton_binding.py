# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Surface behavior of the pyaotriton binding that callers depend on, and that
differs between binding libraries: enum semantics, writes through nested
members, static constants and sequence arguments. Needs no kernel images.
"""

import pytest
import torch  # pyaotriton needs libtorch_cpu loaded first

import pyaotriton
from pyaotriton import T0, T4, DType, Stream, hipError_t
from pyaotriton import v3 as v3_api
from pyaotriton.v3 import flash as v3


def test_enums_behave_like_ints():
    assert int(DType.kFloat32) == 1
    assert DType.kFloat32 == 1
    assert int(hipError_t.hipErrorIllegalAddress) == 700
    # export_values() puts the members in the module namespace
    assert pyaotriton.kFloat16 == DType.kFloat16
    assert pyaotriton.hipSuccess == hipError_t.hipSuccess


def test_hip_calls_return_enum_members():
    assert pyaotriton.hipGetLastError() == hipError_t.hipSuccess
    assert pyaotriton.hipDeviceSynchronize() == hipError_t.hipSuccess


@pytest.mark.parametrize('seq', [tuple, list])
def test_tensor_view_takes_sequences(seq):
    view = T4(0x10, seq((1, 2, 3, 4)), seq((24, 12, 4, 1)), DType.kFloat16)
    assert view.sizes == [1, 2, 3, 4]
    assert view.strides == [24, 12, 4, 1]
    assert view.dtype == DType.kFloat16
    assert (view.size(2), view.stride(1)) == (3, 12)


def test_tensor_view_rejects_wrong_rank():
    with pytest.raises(TypeError):
        T4(0, (1, 2, 3), (1, 2, 3, 4), DType.kFloat16)


def test_rank0_view():
    view = T0(0x20, DType.kInt32)
    assert view.sizes == []
    assert view.dtype == DType.kInt32


def test_writes_reach_nested_members():
    # A copy-returning getter would make these writes vanish, and a varlen
    # request would silently run as dense.
    params = v3.attn_fwd_params()
    params.varlen_bits.qmode.stacked = 1
    params.varlen_bits.kmode.length = 2
    params.varlen_bits.lse_layout = 1
    assert params.varlen_bits.qmode.stacked == 1
    assert params.varlen_bits.kmode.length == 2
    assert params.varlen_bits.lse_layout == 1
    assert v3.varlen_to_wire(params.varlen_bits) != 0


def test_static_constants():
    assert v3.attn_fwd_params.kVersion > 0
    assert v3.attn_bwd_params.kVersion > 0
    assert v3.VarlenStacked.BHSD != v3.VarlenStacked.THD
    assert v3.OpAttnFwdBackend.by_index[0] == 'triton'
    assert v3.OpAttnFwdBackend.Max == len(v3.OpAttnFwdBackend.by_index)


def test_params_hold_tensor_views():
    params = v3.attn_fwd_params()
    params.Q = T4(0x10, (1, 2, 3, 4), (24, 12, 4, 1), DType.kFloat16)
    assert params.Q.sizes == [1, 2, 3, 4]


def test_attn_options_defaults():
    options = v3.attn_options()
    assert options.force_backend_index == -1
    assert options.deterministic is False


def test_stream_is_required():
    params = v3.attn_fwd_params()
    with pytest.raises(TypeError):
        v3.attn_fwd(params, v3.attn_fwd_params.kVersion)
    # An empty parameter set is rejected by the library, not by Python
    result = v3.attn_fwd(params, v3.attn_fwd_params.kVersion, Stream(), None)
    assert result != hipError_t.hipSuccess


def test_submodules_are_registered():
    import sys
    for name in ('v2', 'v2.flash', 'v3', 'v3.flash', 'lazy_tensor'):
        assert f'pyaotriton.{name}' in sys.modules


tuning_only = pytest.mark.skipif(not hasattr(v3_api, 'KernelControl'),
                                 reason='needs AOTRITON_BUILD_FOR_TUNING=ON')


@tuning_only
def test_fine_control_writes_reach_the_options():
    # level_kernel.py does `options.kernel_fine_control[slot].hsaco_index = n`
    options = v3.attn_options()
    slot = v3.attn_options.KernelSlot.attn_fwd
    control = options.kernel_fine_control[slot]
    control.hsaco_index = 5
    control.control_bits = v3_api.KernelControl.Default | v3_api.KernelControl.Manual
    again = options.kernel_fine_control[slot]
    assert (again.hsaco_index, again.control_bits) == (5, v3_api.KernelControl.Manual)
    assert len(options.kernel_fine_control) == int(v3.attn_options.KernelSlot.MaxKernels)


@tuning_only
def test_fine_control_index_out_of_range():
    with pytest.raises(IndexError):
        v3.attn_options().kernel_fine_control[999]


@tuning_only
def test_empty_kernel_image():
    control = v3_api.KernelControl()
    assert control.get_kernel_image() == b''
    assert control.image_size == 0
