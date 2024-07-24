from collections import namedtuple
from aotriton_flash import hipError_t, CppTuneSpecialKernelIndex
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
    num_of_subkernels, outs = fn(is_testing=True)
    for ko in outs:
        if ko.hip_status != hipError_t.hipSuccess:
            # print(f'{ret=}', file=sys.stderr, flush=True)
            return float('inf'), ko.hip_status
    torch.cuda.synchronize()
    valret = validator(outs)
    print(f'{valret=}', flush=True)
    if not valret:
        # assert False
        return float('inf'), outs[0].hip_status
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

KernelOutput = namedtuple('KernelOutput', ['hip_status', 'output_tensors'])
AutotuneResult = namedtuple('AutotuneResult', ['kernel_index', 'time', 'psels', 'copts'])

class CppTuneWapper(object):
    def __init__(self, factory, sub_accessor=None):
        self._extargs = factory()
        self._sub_accessor = sub_accessor
        self._current_sub = 0

    def set_current_sub(self, i):
        self._current_sub = i

    @property
    def current_extargs(self):
        if self._sub_accessor is None:
            return self._extargs
        return self._sub_accessor(self._extargs, self._current_sub)

    @property
    def force_kernel_index(self):
        return self.current_extargs.force_kernel_index

    @force_kernel_index.setter
    def force_kernel_index(self, ki):
        self.current_extargs.force_kernel_index = ki

    @property
    def total_number_of_kernels(self):
        return self.current_extargs.total_number_of_kernels

    @property
    def selected_kernel_psels(self):
        return self.current_extargs.selected_kernel_psels

    @property
    def selected_kernel_copts(self):
        return self.current_extargs.selected_kernel_copts

    @property
    def capi_object(self):
        return self._extargs

# In case extargs.total_number_of_kernels never returns a positive number
# Thus the number does not need to be too large since total_number_of_kernels
# will eventually get updated by the output of extargs
CPP_AUTOTUNE_MAX_KERNELS = 20

''' Use extargs to profile pre-compiled GPU kernels

@param extarg_factory: factory to construct extargs, can be class
@param kernel_func: function that launch GPU kernels, taking the following form
    @param extargs: CppTuneWapper
    @param is_testing: run the kernel for testing purpose rather than measuring its performance.
        Lots of preventive/defensive methods should be enabled if is_testing is
        true. Notably:
            1. Filling the output tensors with nan;
            2. Running the GPU kernel in a separate process to avoid possible
               GPU segfault.
    @returns ret: kernel_func may profile multiple kernels simultaneously, so
                  the return value is a tuple and follows
        @0 number of kernels
        @1 list of KernelOutput
@param validator: validator
'''
def cpp_autotune(extarg_factory, kernel_func, validator, *, tqdm_position=None, tqdm_prefix=''):
    assert validator is not None
    extargs = CppTuneWapper(extarg_factory)
    return cpp_autotune_sub_kernel(extargs, kernel_func, validator,
                                   tqdm_position=tqdm_position,
                                   tqdm_prefix=tqdm_prefix)

def cpp_autotune_sub_kernel(extargs, kernel_func, validator, *, tqdm_position=None, tqdm_prefix=''):
    kernel_index = 0
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
            print(f'Benchmarking with {kernel_index=}. Time {t} {hip_status=}')
        else:
            print(f'Benchmarking with {kernel_index=} out of {extargs.total_number_of_kernels}. Time {t} {hip_status=}')
        '''
        # assert extargs.total_number_of_kernels > 0
        if math.isinf(t):
            if hip_status == hipError_t.hipErrorInvalidImage:
                noimage += 1
            else:
                failed += 1
        else:
            if extargs.total_number_of_kernels > 0:
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

'''
Tuning function for API with multiple kernels
'''
def cpp_autotune_multikernel(extarg_factory, sub_extarg_accessor,
                             subkernel_names, kernel_func,
                             validators, *, tqdm_position=None, tqdm_prefix=''):
    extargs_with_subs = CppTuneWapper(extarg_factory, sub_extarg_accessor)
    num_of_subkernels = len(subkernel_names)
    def reset_kernel_index_to_skip():
        for i in range(num_of_subkernels):
            sub_extarg_accessor(extargs_with_subs.capi_object, i).force_kernel_index = CppTuneSpecialKernelIndex.kSkipGPUCall
    all_ret = []
    for sub_index, cur_name, cur_validator in zip(range(num_of_subkernels), subkernel_names, validators):
        print(f'Tuning sub {cur_name}')
        more_prefix = f'sub {cur_name} {sub_index:02d}/{num_of_subkernels:02d}' if tqdm_prefix else ''
        reset_kernel_index_to_skip()
        extargs_with_subs.set_current_sub(sub_index)
        ret = cpp_autotune_sub_kernel(extargs_with_subs, kernel_func, cur_validator,
                                      tqdm_position=tqdm_position,
                                      tqdm_prefix=tqdm_prefix+more_prefix)
        all_ret.append((cur_name, ret))
    return all_ret
