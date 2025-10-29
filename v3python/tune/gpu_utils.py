# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import os
from contextlib import contextmanager, ExitStack
import torch
from pyaotriton import (
    get_name_suffix,
    T0,
    T1,
    T2,
    T4,
    DType,
    Stream,
    hipError_t,
    hipGetLastError,
    HipMemory,
    hipDeviceSynchronize,
)
from .utils import asdict_shallow
from .defaults import (
    default_device_type,
    default_device_id,
    default_device_string,
)

def elike(t: torch.Tensor | None) -> torch.Tensor | None:
    return torch.empty_like(t) if t is not None else None

def adiff1(golden: torch.Tensor | None,
           lowp: torch.Tensor | None) -> float | None:
    if golden is None or lowp is None:
        assert lowp is None
        return None
    return torch.max(torch.abs(golden.detach() - lowp.detach())).item()

def adiff2(golden: torch.Tensor | None,
           lowp: torch.Tensor | None) -> float | None:
    if golden is None or lowp is None:
        assert lowp is None
        return None
    return (golden, adiff1(golden, lowp))

def strip_grad_l1(golden: torch.Tensor | None,
                  lowp: torch.Tensor | None) -> float | None:
    if golden is None or lowp is None:
        assert golden is None
        assert lowp is None
        return None
    golden_grad, golden.grad = golden.grad.clone(), None
    lowp_grad, lowp.grad = lowp.grad.clone(), None
    return adiff2(golden_grad, lowp_grad)

def target_fudge_factor(out: torch.Tensor,
                        golden: tuple[torch.Tensor, float]) -> tuple[float, float, float]:
    if golden is None or out is None:
        assert golden is None
        assert out is None
        return None
    golden_out, ref_error = golden
    adiff = adiff1(out, golden_out)
    tft = max(1.0, adiff / ref_error)
    return (tft, adiff, ref_error)

def detach_member_tensors(data_object) -> dict:
    d = asdict_shallow(data_object)
    return { k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in d.items() }

@contextmanager
def device_ctx():
    with ExitStack() as stack:
        r1 = stack.enter_context(torch.device(default_device_string()))
        r2 = stack.enter_context(getattr(torch, default_device_type()).device(default_device_id()))
        yield r1, r2

def do_bench(fn,
             *, warmup=25, rep=100,
             grad_to_none=None,
             quantiles=None,
             fast_flush=True,
             return_mode="mean"):
    """
    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param quantiles: Performance percentile to return in addition to the median.
    :type quantiles: list[float], optional
    :param fast_flush: Use faster kernel to flush L2 cache between measurements
    :type fast_flush: bool, default is True
    :param return_mode: The statistical measure to return. Options are "min", "max", "mean", or "median". Default is "mean".
    :type return_mode: str
    """
    assert return_mode in ["min", "max", "mean", "median"]
    assert hipGetLastError() != hipError_t.hipErrorIllegalAddress

    torch.cuda.synchronize()
    # We maintain a buffer of 1024 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 1024 * 1024 * 1024
    if fast_flush:
        cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(cache_size), dtype=torch.int8, device='cuda')
    torch.cuda.synchronize()

    # Estimate the runtime of the function
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(n_repeat)]
    # Warm-up
    for _ in range(n_warmup):
        fn()
    # Benchmark
    for i in range(n_repeat):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    # Record clocks
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_event, end_event)], dtype=torch.float)
    if quantiles is not None:
        return torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
    return getattr(torch, return_mode)(times).item()

def cast_dtype(dtype):
    assert not dtype.is_complex
    bits = dtype.itemsize * 8
    if dtype.is_floating_point:
        maintype = 'Float' if 'bfloat' not in str(dtype) else 'BFloat'
    else:
        maintype = 'Int' if 'uint' not in str(dtype) else 'UInt'
    typename = f'k{maintype}{bits}'
    return getattr(DType, typename)

def _do_mk_aotensor(q, if_empty_then_like=None, force_data_ptr=None):
    rank = len(q.shape) if q is not None else len(if_empty_then_like.shape)
    def lazy_data_ptr():
        return q.data_ptr() if force_data_ptr is None else force_data_ptr
    if q is not None and len(q.shape) == 1 and q.numel() in [0, 1]:
        return T0(lazy_data_ptr(), cast_dtype(q.dtype))
    elif rank == 1:
        klass = T1
    elif rank == 2:
        klass = T2
    elif rank == 4:
        klass = T4
    else:
        assert False, f'Unsupported tensor rank {rank}, shape {q.shape}'
    if q is None:
        return klass(0, [0] * rank, [0] * rank, cast_dtype(if_empty_then_like.dtype))
    if q is not None:
        assert q.stride(-1) == 1, "AOTriton assumes the last stride of Tensors be 1"
    return klass(lazy_data_ptr(), tuple(q.size()), q.stride(), cast_dtype(q.dtype))

def mk_aotensor_cputorch(q, if_empty_then_like=None):
    if q is None or q.device.type != 'cpu':
        return _do_mk_aotensor(q, if_empty_then_like=if_empty_then_like), q
    devm = HipMemory()
    nbytes = q.untyped_storage().nbytes()
    devm.alloc(nbytes)
    devm.load_from_host(q.data_ptr(), nbytes)
    qview = _do_mk_aotensor(q,
                            if_empty_then_like=if_empty_then_like,
                            force_data_ptr=devm.get_pointer())
    return qview, devm

def mk_aotensor_cudatorch(q, if_empty_then_like=None):
    return _do_mk_aotensor(q, if_empty_then_like=if_empty_then_like), q

AOTRITON_TORCH_ONLY_USE_CPU = bool(int(os.getenv('AOTRITON_TORCH_ONLY_USE_CPU', default='0')))

if AOTRITON_TORCH_ONLY_USE_CPU:
    mk_aotensor = mk_aotensor_cputorch
    def zero_devm(devm):
        devm.zero_memory()
else:
    mk_aotensor = mk_aotensor_cudatorch
    def zero_devm(devm):
        devm.zero_()

def create_aotensor_like(like_tensor, if_none_then_like=None):
    if like_tensor is None:
        return mk_aotensor_cudatorch(like_tensor, if_none_then_like)
    devm = torch.empty_like(like_tensor)
    return _do_mk_aotensor(devm), devm

# Note: we don't use Enum class because accessing the integer requires using
#       `.value` property, which makes the code verbose.
class CausalType:
    NONE = 0
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2
    WINDOWED = 3

class WindowValue:
    NONE = 0
    TOP_LEFT_ALIGNED = -2147483647       # 0x80000001. Special value for varlen
    BOTTOM_RIGHT_ALIGNED = -2147483646   # 0x80000002. Special value for varlen

def translate_causal(causal, v3_api):
    window_left, window_right = 0, 0
    if isinstance(causal, tuple):
        assert v3_api, 'Only V3_API supports windowed attention (causal = tuple([window_left, window_right]))'
        window_left, window_right = causal
        causal_type = CausalType.WINDOWED
    elif isinstance(causal, bool):
        # causal_type = CausalType.TOP_LEFT if causal else CausalType.NONE
        causal_type = CausalType.WINDOWED if causal else CausalType.NONE
        if causal:
            window_left = WindowValue.BOTTOM_RIGHT_ALIGNED
            window_right = WindowValue.BOTTOM_RIGHT_ALIGNED
    else:
        assert causal in [CausalType.NONE, CausalType.TOP_LEFT, CausalType.BOTTOM_RIGHT]
        assert v3_api, 'CausalType.TOP_LEFT/BOTTOM_RIGHT variant is supported thru windowed attention, which requires V3 API'
        if causal == CausalType.TOP_LEFT:
            causal_type = CausalType.WINDOWED
            window_left = WindowValue.TOP_LEFT_ALIGNED
            window_right = WindowValue.TOP_LEFT_ALIGNED
        elif causal == CausalType.BOTTOM_RIGHT:
            causal_type = CausalType.WINDOWED
            window_left = WindowValue.BOTTOM_RIGHT_ALIGNED
            window_right = WindowValue.BOTTOM_RIGHT_ALIGNED
        else:
            causal_type = causal
    return causal_type, window_left, window_right
