// Copyright © 2023-2025 Advanced Micro Devices, Inc.
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
    params.kernel_on_device = nullptr;
    auto tune_func = autotune_table[arch_number][params.godel_number()];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(params);
    if (!params.kernel_on_device)
        return hipErrorSharedObjectSymbolNotFound;
    return hipSuccess;
}

hipError_t
[[context_class_name]]::launch(const [[param_class_name]]& params, hipStream_t stream) {
    constexpr std::string_view triton_kernel_name { "[[triton_kernel_name]]" };
    auto arch = getArchFromStream(stream);
    hipDeviceptr_t global_scratch = 0;
    [[put_kernel_arguments_on_stack]];
    std::vector<void*> args = { [[let_kernel_arguments]],
                                const_cast<void*>(static_cast<const void*>(&global_scratch)),
    };
    dim3 grid = grid_calculator(params);
#if AOTRITON_BUILD_FOR_TUNING
    return params.kernel_on_device->invoke(triton_kernel_name,
                                           params.package_path,
                                           params.func_name,
                                           params.arch_name,
                                           grid,
                                           args,
                                           peek_kernel_image,
                                           stream);
#else
    return params.kernel_on_device->invoke(triton_kernel_name,
                                           params.package_path,
                                           params.func_name,
                                           params.arch_name,
                                           grid,
                                           args,
                                           stream);
#endif
}

int64_t
[[context_class_name]]::get_arch_number(GpuArch arch) {
    [[get_arch_number_body]];
    return -1;
}

[[define_compiled_in_features]]

namespace autotune {

[[list_of_deduplicated_lut_functions]]

} // namespace autotune

[[context_class_name]]::AutoTuneTableEntry
[[context_class_name]]::autotune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

}
