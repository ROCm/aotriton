from collections import namedtuple
from aotriton_flash import hipError_t, CppTuneSpecialKernelIndex
import json
import sys
import os
import math
from copy import deepcopy
from .datatypes import CPP_AUTOTUNE_MAX_KERNELS

CPPTUNE_DEBUG_FEW_KERNELS = int(os.getenv('CPPTUNE_DEBUG_FEW_KERNELS', default=-1))

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
    # print(f'{num_of_subkernels=}')
    # print(f'{outs=}')
    for ko in outs:
        if ko.hip_status != hipError_t.hipSuccess:
            # print(f'{ret=}', file=sys.stderr, flush=True)
            return float('inf'), -1.0, ko.hip_status
    torch.cuda.synchronize()
    valret, adiff = validator(outs)
    # print(f'{valret=} {outs[0].hip_status=}', flush=True)
    if not valret:
        # assert False
        return float('inf'), adiff, outs[0].hip_status
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
        return ret, adiff, hipError_t.hipSuccess
    return getattr(torch, return_mode)(times).item(), adiff, hipError_t.hipSuccess

KernelOutput = namedtuple('KernelOutput', ['hip_status', 'output_tensors'])
AutotuneResult = namedtuple('AutotuneResult', ['hip_status', 'kernel_index', 'total_number_of_kernels', 'time', 'psels', 'copts'])

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

def cpp_autotune_sub_kernel_gen(extargs, kernel_func, validator, cur_kig):
    if cur_kig.kernel_index >= cur_kig.total_number_of_kernels:
        '''
        CAVEAT: Must run kernel_func at least once.
                Otherwise this may happen:
                    1. Running fwd and bwd tuning;
                    2. Bwd kernel segfaulted;
                    3. Resume the tuning process after skipping the kernel;
                    4. The o tensor is empty/nan and the bwd output becomes garbage.
        '''
        extargs.force_kernel_index = cur_kig.last_success_kernel
        kernel_func(extargs, is_testing=False)
        return
    # print(f'{cur_kig.kernel_index=}')
    while True:
        # max(0, ...) is the defensive approach to prevent -1 slipping into C++ component
        extargs.force_kernel_index = max(0, cur_kig.kernel_index)
        def func(is_testing=False):
            return kernel_func(extargs, is_testing)
        t, adiff, hip_status = do_bench(func, validator=validator, quantiles=(0.5, 0.2, 0.8))
        # assert extargs.total_number_of_kernels > 0
        '''
        Update kig
        '''
        if extargs.total_number_of_kernels > 0:
            cur_kig.total_number_of_kernels = extargs.total_number_of_kernels
            # !!!!!!!!!!!!!!!!!!! DEBUG !!!!!!!!!!!!!!!!!!!!!!!!!!
            if CPPTUNE_DEBUG_FEW_KERNELS > 0:
                cur_kig.total_number_of_kernels = min(CPPTUNE_DEBUG_FEW_KERNELS, extargs.total_number_of_kernels)
        cur_kig.last_adiff = adiff
        if hip_status == hipError_t.hipSuccess and not (isinstance(t, float) and math.isinf(t)):
            cur_kig.last_success_kernel = extargs.force_kernel_index
            cur_kig.passed_kernels += 1
        else:
            if hip_status == hipError_t.hipErrorInvalidImage:
                cur_kig.noimage_kernels += 1
            else:
                cur_kig.failed_kernels += 1
        def safeload(s):
            return json.loads(s) if s else None
        cur_kig.kernel_index = extargs.force_kernel_index
        yield AutotuneResult(hip_status=hip_status,
                             kernel_index=cur_kig.kernel_index,
                             total_number_of_kernels=cur_kig.total_number_of_kernels,
                             time=t,
                             psels=safeload(extargs.selected_kernel_psels),
                             copts=safeload(extargs.selected_kernel_copts))
        cur_kig.kernel_index = extargs.force_kernel_index + 1
        if cur_kig.kernel_index >= cur_kig.total_number_of_kernels:
            break
    # TODO: Report no conf works

''' Use extargs to profile pre-compiled GPU kernels
This is an generator of all tuning results, and yields all results rather than the best one.

@param extarg_factory: factory to construct extargs for the API, can be class
@param sub_extarg_accessor: extract extargs for sub kernels within the API call
@param subkernel_names: names of sub kernels for API call
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
@param validator: validator of each sub kernel
@param kernel_index_progress_dict: dict[subkernel_name, KernelIndexProress].
    Use to track the progress and resume the task if interrupted (e.g. by GPU Hangs)
'''
def cpp_autotune_gen(extarg_factory, sub_extarg_accessor,
                     subkernel_names, kernel_func,
                     validators, *, kernel_index_progress_dict):
    extargs_with_subs = CppTuneWapper(extarg_factory, sub_extarg_accessor)
    num_of_subkernels = len(subkernel_names)
    def reset_kernel_index_to_skip():
        for i in range(num_of_subkernels):
            sub_extarg_accessor(extargs_with_subs.capi_object, i).force_kernel_index = CppTuneSpecialKernelIndex.kSkipGPUCall
    all_ret = []
    # kig_dict = deepcopy(kernel_index_progress_dict)
    kig_dict = kernel_index_progress_dict  # Otherwise kernel progress are local to this function
    for sub_index, cur_name, cur_validator in zip(range(num_of_subkernels), subkernel_names, validators):
        # print(f'Tuning sub {cur_name}')
        reset_kernel_index_to_skip()
        extargs_with_subs.set_current_sub(sub_index)
        cur_kig = kig_dict[cur_name]
        for ret in cpp_autotune_sub_kernel_gen(extargs_with_subs,
                                               kernel_func,
                                               cur_validator,
                                               cur_kig):
            yield cur_name, ret, deepcopy(kig_dict)
