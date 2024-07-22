from collections import namedtuple
from aotriton_flash import hipError_t
import json
import sys
import math
from tqdm import tqdm

def do_bench(fn, *, warmup=25, rep=100,
             grad_to_none=None,
             quantiles=None,
             fast_flush=True,
             return_mode="mean",
             validator=None):
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
    import torch

    torch.cuda.synchronize()
    ret, outs = fn(is_testing=True)
    if ret != hipError_t.hipSuccess:
        # print(f'{ret=}', file=sys.stderr, flush=True)
        return float('inf'), ret
    torch.cuda.synchronize()
    valret = validator(*outs)
    # print(f'{valret=}', flush=True)
    if not valret:
        # assert False
        return float('inf'), ret
    torch.cuda.synchronize()

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    if fast_flush:
        cache = torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
    else:
        cache = torch.empty(int(cache_size), dtype=torch.int8, device='cuda')

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
        ret = torch.quantile(times, torch.tensor(quantiles, dtype=torch.float)).tolist()
        if len(ret) == 1:
            ret = ret[0]
        return ret, hipError_t.hipSuccess
    return getattr(torch, return_mode)(times).item(), hipError_t.hipSuccess

AutotuneResult = namedtuple('AutotuneResult', ['kernel_index', 'time', 'psels', 'copts'])

# cannot be maxint in case extargs.total_number_of_kernels never returns a positive number
CPP_AUTOTUNE_MAX_KERNELS = 200

def cpp_autotune(extarg_klass, kernel_func, validator, *, tqdm_position=None, tqdm_prefix=''):
    assert validator is not None
    kernel_index = 0
    extargs = extarg_klass()
    timings = []
    pbar = None
    failed = 0
    success = 0
    noimage = 0
    total_number_of_kernels = CPP_AUTOTUNE_MAX_KERNELS
    while True:
        extargs.force_kernel_index = kernel_index
        def func(is_testing=False):
            return kernel_func(extargs, is_testing)
        # t = do_bench(func, validator=validator, quantiles=(0.5, 0.2, 0.8))
        t, hip_status = do_bench(func, validator=validator)
        '''
        if kernel_index == 0:
            print(f'Benchmarking with {kernel_index=}. Time {t}')
        else:
            print(f'Benchmarking with {kernel_index=} out of {extargs.total_number_of_kernels}. Time {t}')
        '''
        # assert extargs.total_number_of_kernels > 0
        if math.isinf(t):
            if hip_status == hipError_t.hipErrorInvalidImage:
                noimage += 1
            else:
                failed += 1
        else:
            if extargs.total_number_of_kernels > 0:
                assert extargs.total_number_of_kernels <= CPP_AUTOTUNE_MAX_KERNELS
                total_number_of_kernels = extargs.total_number_of_kernels
            success += 1
            r = AutotuneResult(kernel_index=kernel_index,
                               time=t,
                               psels=json.loads(extargs.selected_kernel_psels),
                               copts=json.loads(extargs.selected_kernel_copts))
            timings.append(r)

        pbar_desc = f'{tqdm_prefix} Pass/Fail/NoImage {success}/{failed}/{noimage}. Last time {t:.2g}'
        if pbar is None and extargs.total_number_of_kernels > 0:
            pbar = tqdm(total=extargs.total_number_of_kernels, unit="configs", position=tqdm_position)
            pbar.set_description(pbar_desc)
        if pbar is not None:
            pbar.set_description(pbar_desc)
            pbar.update(1)

        #     print(f'{r.psels=}')
        #     print(f'{r.copts=}')
        kernel_index += 1
        if kernel_index >= total_number_of_kernels:
            break
    # print(f'cpp_autotune {timings=}')
    ret = min(timings, key=lambda atr:atr.time)
    # print(f'{ret=}')
    if math.isinf(ret.time):
        # with open("/proc/self/maps") as f:
        #     print(f.read(), file=sys.stderr)
        print("ERROR: No configuration works")
    return ret
