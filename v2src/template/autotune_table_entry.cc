// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../shim.[[shim_kernel_name]].h"
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/kernel_cluster.h>
#include <aotriton/cpp_tune.h>
#ifndef NDEBUG // for std::cerr
    #include <iostream>
#endif
#include <string_view>

// [[human_readable_signature]]
#define CURRENT_ENTRY_PUBLIC Autotune_[[shim_kernel_name]]__A[[arch_number]]__F[[godel_number]]

#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(array[0]))

namespace { // Anonymous namespace

using namespace std::literals::string_view_literals;

#if AOTRITON_BUILD_FOR_TUNING
// PSels and Copts in JSON String
[[kernel_psels]];
[[kernel_copts]];
#endif

struct PerfFields {
  [[perf_fields]];
};

static PerfFields image_perf_list [] = {
    [[kernel_image_perfs]]
};

constexpr std::string_view PACKAGE_PATH { R"xyzw([[package_path]])xyzw" };
constexpr std::string_view FUNC_NAME { R"xyzw([[func_name]])xyzw" };
constexpr std::string_view ARCH_NAME { R"xyzw([[arch_name]])xyzw" };

static const char packed_string[] =
[[packed_string]];

// Checksum can be confirmed with `echo -n '<string>' | b2sum -l 64`
// For example:
//   $ echo -n 'amd-gfx110x/flash/attn_fwd/FONLY__^bf16@16,128,False,False,0,0___gfx1100__P__32_32_0_2_False__CO__wave3_warp2_stg1-Gpu-gfx1100' | b2sum -l 64
//   c4b51ee645d79580  -
static AOTRITON_NS::TritonKernelCompactMeta meta_list[] = {
    [[meta_objects]]
};

static constexpr int kTotalNumKernels = ARRAY_SIZE(meta_list);

static AOTRITON_NS::TritonKernelCluster<kTotalNumKernels> kernel_cluster(meta_list, packed_string);

static [[lut_dtype]] lut[[lut_shape]] =
[[lut_data]]
;

}; // End of anonymous namespace

namespace AOTRITON_NS::v2::[[kernel_family_name]]::autotune {

// using AOTRITON_NS::v2::[[kernel_family_name]]::[[param_class_name]];

void CURRENT_ENTRY_PUBLIC([[param_class_name]]& params, int mod_number) {
#if AOTRITON_BUILD_FOR_TUNING
    int preferred_index = params._has_preferred_kernel;
    params._total_number_of_kernels = kTotalNumKernels;
    if (preferred_index != -1) {
        if (preferred_index >= kTotalNumKernels)
            return ;
        params.kernel_on_device = kernel_cluster.get(preferred_index);
        params._preferred_kernel_psels = kernel_psels[preferred_index];
        params._preferred_kernel_copts = kernel_copts[preferred_index];
        const auto& perf = image_perf_list[preferred_index];
        [[perf_field_assignment]];
        return ;
    }
#endif
    auto kernel_index = [[deduplicated_lut_function]](params, mod_number, lut);
    if (kernel_index < 0) {
      return ;
    }
    params.kernel_on_device = kernel_cluster.get(kernel_index);
    params.pp_args_index = [[deduplicated_pp_args_function_index]];
    params.package_path = PACKAGE_PATH;
    params.func_name = FUNC_NAME;
    params.arch_name = ARCH_NAME;
#ifndef NDEBUG
    std::cerr << __FILE__ << " kernel_index = " << int(kernel_index) << std::endl;
#endif
    const auto& perf = image_perf_list[kernel_index];
    [[perf_field_assignment]];
}

#undef CURRENT_ENTRY_PUBLIC
#undef mangle
#undef smangle
}
