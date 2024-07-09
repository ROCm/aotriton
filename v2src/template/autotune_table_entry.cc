// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#define INCBIN_PREFIX g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE

#define mangle(x) g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_ ## x ## _data
#define smangle(x) g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_ ## x ## _size

#include "../shim.[[shim_kernel_name]].h"
#include <aotriton/_internal/triton_kernel.h>
#include <incbin.h>
#include <iostream>

// [[human_readable_signature]]
#define CURRENT_ENTRY_PUBLIC Autotune_[[shim_kernel_name]]__A[[arch_number]]__F[[godel_number]]

[[incbin_kernel_images]];

#if defined(NDEBUG) || AOTRITON_BUILD_FOR_TUNING
[[incbin_kernel_names]];
#endif

#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(array[0]))

namespace { // Anonymous namespace

#if AOTRITON_BUILD_FOR_TUNING
static constexpr int incbin_num_kernels = ARRAY_SIZE(incbin_kernel_names);
#endif

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

aotriton::TritonKernel image_list [] = {
    [[kernel_image_objects]]
};

[[lut_dtype]] lut[[lut_shape]] = [[lut_data]];

}; // End of anonymous namespace

namespace aotriton::v2::[[kernel_family_name]]::autotune {

// using aotriton::v2::[[kernel_family_name]]::[[param_class_name]];

void CURRENT_ENTRY_PUBLIC::operator()([[param_class_name]]& params) {
#if AOTRITON_BUILD_FOR_TUNING
    int preferred_index = params._has_preferred_kernel;
    params._total_number_of_kernels = incbin_num_kernels;
    if (preferred_index >= 0) {
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
