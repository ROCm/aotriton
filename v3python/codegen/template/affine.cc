// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "affine.[[affine_kernel_name]].h"
#include <aotriton/_internal/kernel_cluster.h>
#include <aotriton/util.h>
#include <tuple>
#include <iostream>
[[includes]]

namespace AOTRITON_NS::v3::[[kernel_family_name]] {

#if [[shared_iface]]
using AOTRITON_NS::v3::[[shared_iface_family]]::[[param_class_name]];
#endif

int64_t [[context_class_name]]::godel_number() const
{
    int64_t sum = 0;
    const auto& args = *params;
[[godel_number_body]]
    return sum;
}

hipError_t
[[context_class_name]]::lookup_optimal(Gpu gpu) {
    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    const char* reject_reason = check_inputs_are_supported();
    if (reject_reason) {
#ifndef NDEBUG
        std::cerr << "Unsupported inputs for backend "
                  << "[[context_class_name]]"
                  << " reason: "
                  << reject_reason
                  << std::endl;
#endif
        return hipErrorPeerAccessUnsupported;
    }
    calculate_residual_func_fields();
    kernel_on_device = nullptr;
    // Unlike Triton's autotune_table
    // Affine kernel uses entries from "capability_table", which validate if
    // input is supported.
    auto number = godel_number();
    if (number < 0) {
#ifndef NDEBUG
        std::cerr << "Unsupported inputs for backend "
                  << "[[context_class_name]]"
                  << " reason: cannot assign godel number "
                  << std::endl;
#endif
        return hipErrorPeerAccessUnsupported;
    }
    auto capability_validator = capability_table[arch_number][number];
    if (!capability_validator) {
#ifndef NDEBUG
        std::cerr << "Unsupported inputs for backend "
                  << "[[context_class_name]]"
                  << " reason: capability table has no entry for godel number "
                  << number
                  << std::endl;
#endif
        return hipErrorPeerAccessUnsupported;
    }
    // capability_validator is responsible to
    // 1. return hipErrorPeerAccessUnsupported when kernel cannot handle inputs
    //    (Usually not required, can be identified with residual choices)
    // 2. assign selected_pp_args
    // 3. assign affine_kernel_name/package_path/function_name/arch_name
    // 4. assign kernel_on_device
    return capability_validator(*this, mod_number);
}

std::tuple<int, int>
[[context_class_name]]::get_archmod_number(Gpu gpu) {
    [[get_archmod_number_body]];
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}


hipError_t
[[context_class_name]]::launch(hipStream_t stream) const {
    DirectKernelArguments direct_args;
    auto [grid, block] = (*this.*selected_pp_args)(direct_args);
    return kernel_on_device->direct_invoke(affine_kernel_function_name,
                                           package_path,
                                           affine_kernel_function_name,
                                           arch_name,
                                           grid,
                                           block,
                                           &direct_args,
                                           sizeof_selected_args,
                                           stream);
}

namespace {

// Kernels from ALL arches go here.
AOTRITON_NS::TritonKernelCompactMeta meta_list[] = {
    [[meta_cos]]
};
#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(array[0]))
constexpr int kTotalNumKernels = ARRAY_SIZE(meta_list);
#undef ARRAY_SIZE
const char packed_string[] =
[[kernel_co_name_packed_string]];

AOTRITON_NS::TritonKernelCluster<kTotalNumKernels>
kernel_cluster(meta_list, packed_string);

[[validator_defs]]
}

[[context_class_name]]::CapabilityTableEntry
[[context_class_name]]::capability_table[][ [[number_of_functionals_with_residuals]] ] = {
[[capability_table_entries]]
};

}
