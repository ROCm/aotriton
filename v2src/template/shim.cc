// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "shim.[[shim_kernel_name]].h"
#include <aotriton/util.h>

namespace AOTRITON_NS::v2::[[kernel_family_name]] {

int64_t [[param_class_name]]::godel_number() const
{
    int64_t sum = 0;
[[godel_number_body]]
    return sum;
}

hipError_t
[[context_class_name]]::lookup_optimal([[param_class_name]]& params, GpuArch arch) {
    int64_t arch_number = get_arch_number(arch);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    params.selected_kernel = nullptr;
    auto tune_func = autotune_table[arch_number][params.godel_number()];
    tune_func(params);
    if (!params.selected_kernel)
        return hipErrorSharedObjectSymbolNotFound;
    return hipSuccess;
}

hipError_t
[[context_class_name]]::launch(const [[param_class_name]]& params, hipStream_t stream) {
    auto arch = getArchFromStream(stream);
    hipDeviceptr_t global_scratch = 0;
    [[put_kernel_arguments_on_stack]];
    std::vector<void*> args = { [[let_kernel_arguments]],
                                const_cast<void*>(static_cast<const void*>(&global_scratch)),
    };
    dim3 grid = grid_calculator(params);
    return params.selected_kernel->invoke("[[triton_kernel_name]]", grid, args, stream);
}

int64_t
[[context_class_name]]::get_arch_number(GpuArch arch) {
    [[get_arch_number_body]];
    return -1;
}

[[define_compiled_in_features]]

[[context_class_name]]::AutoTuneTableEntry
[[context_class_name]]::autotune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

}
