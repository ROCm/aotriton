// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../shim.[[shim_kernel_name]].h"
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/cpp_tune.h>
#include <incbin.h>
#include <iostream>

// [[human_readable_signature]]
#define CURRENT_ENTRY_PUBLIC Autotune_[[shim_kernel_name]]__A[[arch_number]]__F[[godel_number]]

#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(array[0]))

namespace { // Anonymous namespace

#if AOTRITON_BUILD_FOR_TUNING
// PSels and Copts in JSON String
[[kernel_psels]];
[[kernel_copts]];
#endif

struct PerfFields {
  [[perf_fields]];
};

PerfFields image_perf_list [] = {
    [[kernel_image_perfs]]
};

const char* PACKAGE_PATH = [[package_path]];

AOTRITON_NS::TritonKernel image_list [] = {
    [[kernel_image_objects]]
};

#if AOTRITON_BUILD_FOR_TUNING
static constexpr int total_num_kernels = ARRAY_SIZE(image_list);
#endif

[[lut_dtype]] lut[[lut_shape]] = [[lut_data]];

}; // End of anonymous namespace

namespace AOTRITON_NS::v2::[[kernel_family_name]]::autotune {

// using AOTRITON_NS::v2::[[kernel_family_name]]::[[param_class_name]];

void CURRENT_ENTRY_PUBLIC::operator()([[param_class_name]]& params) {
#if AOTRITON_BUILD_FOR_TUNING
    int preferred_index = params._has_preferred_kernel;
    params._total_number_of_kernels = incbin_num_kernels;
    if (preferred_index != -1) {
        if (preferred_index >= incbin_num_kernels)
            return ;
        params.selected_kernel = &image_list[preferred_index];
        params._debug_kernel_name = incbin_kernel_names[preferred_index];
        params._preferred_kernel_psels = kernel_psels[preferred_index];
        params._preferred_kernel_copts = kernel_copts[preferred_index];
        const auto& perf = image_perf_list[preferred_index];
        [[perf_field_assignment]];
        return ;
    }
#endif
    [[binning_autotune_keys]]
    auto kernel_index = lut[[binned_indices]];
    if (kernel_index < 0) {
      return ;
    }
    params.selected_kernel = &image_list[kernel_index];
#ifndef NDEBUG
    std::cerr << __FILE__ << " kernel_index = " << int(kernel_index) << std::endl;
    params._debug_kernel_name = incbin_kernel_names[kernel_index];
#endif
    const auto& perf = image_perf_list[kernel_index];
    [[perf_field_assignment]];
}

#undef CURRENT_ENTRY_PUBLIC
#undef mangle
#undef smangle
}
