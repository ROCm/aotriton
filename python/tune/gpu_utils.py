# Copyright © 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU helpers for the tuning workers.

CAVEAT for anything that prints here: these routines run inside the `testrun`
worker subprocess, whose **stdout is a wire protocol, not a log**.
`ExaidProxy.readinfo()` (exaid.py) reads that stdout a line at a time and
treats the first whitespace-separated token as follows:

  * ``OK`` / ``OK <json>`` -- the reply it is waiting for.
  * ``OVERHEATING: ...``   -- forwarded to the log; it keeps reading.
  * anything else          -- raises ``ExaidSubprocessNotOK``, failing the task.

Two rules follow, and both have already been learned the hard way:

  * **Print diagnostics to ``sys.stderr``**, the way testrun.py does. A bare
    ``print()`` of a warning does not merely add noise -- it aborts whatever
    the worker was asked to do.
  * **Keep the ``OVERHEATING:`` prefix on the cooldown lines, and keep
    emitting one per poll** rather than only after a long wait. `readinfo()`
    resets its read timeout on every line it accepts; `probe()` reads with the
    default 10s timeout and `benchmark()` with 30s, so a worker that goes
    quiet while waiting out a cooldown is killed as unresponsive.

Note this constrains print() only. Raising is fine: exceptions surface through
the worker's exit status and stderr, not through the protocol stream.
"""

import sys
import os
import time
import shutil
from contextlib import contextmanager, ExitStack
from pathlib import Path
import torch

# Import amdsmi with path handling (following amdsmi_cli.py practice)
# Find amd-smi location and add its directory to sys.path
import amdsmi

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

def record_early_reject(tff_result: tuple[float, float, float] | None) -> tuple[float, float, float]:
    if tff_result is None:
        return None
    from pyaotriton import hipError_t
    sentinel = -int(hipError_t.hipErrorPeerAccessUnsupported)
    tft, _adiff, ref_error = tff_result
    return (tft, sentinel, ref_error)

def detach_member_tensors(data_object) -> dict:
    d = asdict_shallow(data_object)
    return { k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in d.items() }

_total_memory_gb = None

def _bdf_of(device_id):
    """PCI BDF of a HIP device, as AMD-SMI spells it (``0000:65:00.0``).

    Sourced from HIP via torch rather than from AMD-SMI: torch.cuda device
    properties come from hipGetDeviceProperties and cost no AMD-SMI call at
    all, so resolving a device never has to enumerate the other GPUs.

    This is also the only mapping that survives ROCR_VISIBLE_DEVICES /
    HIP_VISIBLE_DEVICES. AMD-SMI's own ``hip_id`` is a *global* index that
    ignores the mask, while our device ids are masked ones, so pairing the
    two picks the wrong GPU whenever a mask is set (as SLURM does).
    """
    p = torch.cuda.get_device_properties(device_id)
    return f'{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0'

@contextmanager
def _amdsmi_ctx(device_id=None):
    """Initialize AMD-SMI, yield GPU handle(s), and shut down on exit.

    Yields the handle of `device_id` alone, or the list of every handle when
    `device_id` is None. The single-device form resolves by BDF, so it never
    calls amdsmi_get_processor_handles() and never asks the other GPUs
    anything.

    Handles must not outlive the context. Nesting one context inside another
    is safe, though: amdsmi_init()/amdsmi_shut_down() are refcounted, so the
    inner exit does not invalidate the outer one's handles. That is what lets
    get_total_memory_from_amdsmi() run while _own_amdsmi_device() holds a
    context open. Tearing down the last context does invalidate every handle
    taken from it -- reads then fail with AMDSMI_STATUS_NOT_INIT.
    """
    amdsmi.amdsmi_init()
    try:
        if device_id is None:
            yield amdsmi.amdsmi_get_processor_handles()
        else:
            yield amdsmi.amdsmi_get_processor_handle_from_bdf(_bdf_of(device_id))
    finally:
        amdsmi.amdsmi_shut_down()

def get_total_memory_from_amdsmi():
    """Get total GPU memory in GB from AMD-SMI."""
    global _total_memory_gb
    if _total_memory_gb is not None:
        return _total_memory_gb

    try:
        with _amdsmi_ctx() as devices:
            vram_cap = -1
            for device in devices:
                vram_usage = amdsmi.amdsmi_get_gpu_vram_usage(device)
                total_memory = vram_usage['vram_total'] / 1024  # amdsmi reports MB -> GB
                vram_cap = min(vram_cap, total_memory) if vram_cap > 0 else total_memory
        if vram_cap <= 0:
            # No device answered (an empty handle list leaves the sentinel in
            # place). Report failure instead of memoizing -1: callers test for
            # None, and a negative cap would clamp every shape to the minimum.
            return None
        _total_memory_gb = vram_cap
        return vram_cap
    except Exception:
        return None

# Junction (a.k.a. hotspot) is what we want to throttle on, but not every ASIC
# implements it. gfx1151 exposes an edge sensor only and answers junction with
# AMDSMI_STATUS_NOT_SUPPORTED, which used to abort tuning outright; gfx942 is
# the mirror image, implementing junction but not edge. So the sensor has to be
# probed per device instead of assumed.
_TEMP_SENSORS = (amdsmi.AmdSmiTemperatureType.JUNCTION,
                 amdsmi.AmdSmiTemperatureType.EDGE)

def _pick_temp_sensor(handle, device_id):
    """First sensor in _TEMP_SENSORS that this GPU actually implements."""
    for sensor in _TEMP_SENSORS:
        try:
            amdsmi.amdsmi_get_temp_metric(handle, sensor,
                                          amdsmi.AmdSmiTemperatureMetric.CURRENT)
        except amdsmi.AmdSmiLibraryException as e:
            if e.get_error_code() != amdsmi.amdsmi_wrapper.AMDSMI_STATUS_NOT_SUPPORTED:
                raise
            continue
        if sensor is not _TEMP_SENSORS[0]:
            # stderr, never stdout. Everything here can run inside the testrun
            # worker, whose stdout is the exaid wire protocol: ExaidProxy
            # .readinfo() forwards `OVERHEATING:` lines and raises
            # ExaidSubprocessNotOK on anything else that is not `OK`. A
            # diagnostic on stdout would therefore kill the job -- and this one
            # fires precisely on the gfx1151 parts the fallback exists to keep
            # running.
            print(f'WARNING: GPU HIP ID {device_id} does not implement the '
                  f'{_TEMP_SENSORS[0].name} temperature sensor, '
                  f'falling back to {sensor.name}', flush=True, file=sys.stderr)
        return sensor
    # Loud on purpose: silently skipping the wait would let a hot GPU cook,
    # and bogus thermals surface later as inexplicable tuning results.
    raise RuntimeError(f'GPU HIP ID {device_id} implements none of the '
                       f'temperature sensors {[s.name for s in _TEMP_SENSORS]}')

_amdsmi_stack = None      # keeps AMD-SMI alive for _amdsmi_handle below
_amdsmi_device_id = None
_amdsmi_handle = None
_amdsmi_sensor = None

def _own_amdsmi_device(device_id):
    """(handle, sensor) for `device_id`, keeping AMD-SMI open between calls.

    wait_gpu_temperature() runs on every device_ctx() entry and polls every
    5s while overheating, so re-entering the context per reading would pay an
    amdsmi_init() each time, and re-probing the sensor would pay a failed
    query on every ASIC that lacks a junction sensor. Only this device is held
    onto; the context is torn down and rebuilt if a different device_id shows
    up.
    """
    global _amdsmi_stack, _amdsmi_device_id, _amdsmi_handle, _amdsmi_sensor
    if _amdsmi_stack is not None:
        if _amdsmi_device_id == device_id:
            return _amdsmi_handle, _amdsmi_sensor
        _amdsmi_stack.close()
        _amdsmi_stack = _amdsmi_device_id = _amdsmi_handle = _amdsmi_sensor = None

    stack = ExitStack()
    handle = stack.enter_context(_amdsmi_ctx(device_id))
    try:
        sensor = _pick_temp_sensor(handle, device_id)
    except BaseException:
        stack.close()
        raise
    _amdsmi_stack, _amdsmi_device_id = stack, device_id
    _amdsmi_handle, _amdsmi_sensor = handle, sensor
    return handle, sensor

def _get_temperature_amdsmi(amdsmi_dev, sensor):
    """Read GPU temperature from an AMD-SMI handle held by an open context."""
    return amdsmi.amdsmi_get_temp_metric(
        amdsmi_dev,
        sensor,
        amdsmi.AmdSmiTemperatureMetric.CURRENT
    )

def wait_gpu_temperature(device_id=None, threshold=85.0):
    """Wait until GPU temperature drops below threshold.

    Reports on every poll rather than only once the wait gets long. The
    `OVERHEATING:` prefix is a wire protocol: ExaidProxy.readinfo() forwards
    such lines to the log and keeps reading instead of failing, and each one
    resets its read timeout. probe() reads with the default 10s timeout and
    benchmark() with 30s, so staying quiet through a cooldown would have the
    parent kill a worker that is only waiting for the GPU to cool.
    """
    if device_id is None:
        device_id = default_device_id()

    # Use AMD-SMI directly to avoid HIP ID vs AMD-SMI ID confusion
    amdsmi_dev, sensor = _own_amdsmi_device(device_id)
    temp = _get_temperature_amdsmi(amdsmi_dev, sensor)

    if temp <= threshold:
        return

    start_time = time.time()
    while temp > threshold:
        elapsed = time.time() - start_time
        print(f"OVERHEATING: GPU HIP ID {device_id} TEMP. {temp} "
              f"ELAPSED. {int(elapsed)}s", flush=True)
        time.sleep(5)
        temp = _get_temperature_amdsmi(amdsmi_dev, sensor)
        if temp is None:
            break
    print(f"OVERHEATING: EXIT GPU HIP ID {device_id} TEMP. {temp} "
          f"ELAPSED. {int(time.time() - start_time)}s", flush=True)

@contextmanager
def device_ctx():
    with ExitStack() as stack:
        r1 = stack.enter_context(torch.device(default_device_string()))
        r2 = stack.enter_context(getattr(torch, default_device_type()).device(default_device_id()))
        wait_gpu_temperature()
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

# NOTE: CausalType / WindowValue / translate_causal moved to
# modules/flash/tune/causal.py (modular-tune.md §3b/step 12) -- this module
# is family-neutral and causal-mask translation is flash-specific. Import
# from `aotriton.tune.registry.load_family_tune('flash').causal` (or, from
# within modules/flash/tune/, `from .causal import ...`) instead.
