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
    // Slim Affine Kernel does not use Godel Numbered Functional system
    return -1;
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
    return hipSuccess;
}

// Users of Slim Affine Kernel are responsible to implement launch() function
// hipError_t [[context_class_name]]::launch(hipStream_t stream) const;

}
