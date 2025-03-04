#!/usr/bin/env python
# Copyright © 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

# In case extargs.total_number_of_kernels never returns a positive number
# Thus the number does not need to be too large since total_number_of_kernels
# will eventually get updated by the output of extargs
CPP_AUTOTUNE_MAX_KERNELS = 20

'''
KernelIndexProress.kernel_index has to be -1

KernelIndexProress.kernel_index defaults to 0 is ambiguous: kernel 0 may or may
not be tuned.

The ambiguous comes from the dual use of KernelIndexProress object. It
is the argument for GPU worker to resume a interrupted task (either by
user inputs or by GPU hangs), as well as the data record of tuned
kernels. Hence the precise meaning of kernel_index actually depends on
who stored/generated this object.

This change tries to erase the ambiguity by assigning -1 as "no kernels
are tuned right now" (used by incoming arguments), which is the default
for newly generated KernelIndexProress objects.

Either resuming from json (by source) or state tracker (by manager),
the instigator should calculate the index of the kernel index that should
be tuned, and pass the calculated KernelIndexProress argument to GPU worker.

GPU worker is not responsible to guess where to start the tuning.
'''
@dataclass
class KernelIndexProress:
    kernel_index : int = -1
    last_success_kernel : int = None
    last_adiff : float = -1.0
    total_number_of_kernels: int = CPP_AUTOTUNE_MAX_KERNELS
    passed_kernels : int = 0
    failed_kernels : int = 0
    vspill_kernels : int = 0
    noimage_kernels : int = 0
    uncertain_errors : int = 0
    best_adiffs : 'float | list[float] | None' = None

@dataclass
class TuningResult:
    tup: tuple
    profiled_kernel_name: str = None
    perf_number: list = None
    kig_dict : dict[str, KernelIndexProress] = None
