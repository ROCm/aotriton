import sys
from dataclasses import fields

def parse_python(line: str) -> dict:
    args = line.split(';')
    d = {}
    for assignment in args:
        # print(f'{assignment=}', file=sys.stderr)
        k, v = assignment.split('=', maxsplit=1)
        d[k] = eval(v)
    return d

def asdict_shallow(obj) -> dict:
    return {field.name: getattr(obj, field.name) for field in fields(obj)}

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
    import torch
    from pyaotriton import hipGetLastError, hipError_t
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
