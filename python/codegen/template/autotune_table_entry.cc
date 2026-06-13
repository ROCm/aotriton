// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../shim.[[shim_kernel_name]].h"
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/kernel_cluster.h>
#include <aotriton/cpp_tune.h>
#include <string_view>
#ifndef NDEBUG
#include <iostream>
#include <assert.h>
#endif

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

// u8R generates char8_t which is poorly supported almost everywhere.
// TODO: deduplicate FLATZIP_PATH via shim.cc packed_string
constexpr pstring_view FLATZIP_PATH
#if defined(_WIN32)
{ LR"xyzw([[flatzip_path]])xyzw" };
#else
{ R"xyzw([[flatzip_path]])xyzw" };
#endif
constexpr std::string_view AKS2_ENTRY  { R"xyzw([[aks2_entry]])xyzw"  };
constexpr std::string_view FUNC_NAME   { R"xyzw([[func_name]])xyzw"   };
constexpr std::string_view ARCH_NAME { R"xyzw([[arch_name]])xyzw" };

// Checksum can be confirmed with `echo -n '<string>' | b2sum -l 64`
// For example:
//   $ echo -n 'amd-gfx110x/flash/attn_fwd/FONLY__^bf16@16,128,False,False,0,0___gfx1100__P__32_32_0_2_False__CO__wave3_warp2_stg1-Gpu-gfx1100' | b2sum -l 64
//   c4b51ee645d79580  -
static AOTRITON_NS::TritonKernelCompactMeta meta_list[] = {
    [[meta_hsacos]]
};

static constexpr int kTotalNumKernels = ARRAY_SIZE(meta_list);

using AOTRITON_NS::v3::[[kernel_family_name]]::autotune::[[shim_kernel_name]]_packed_string;

static AOTRITON_NS::TritonKernelCluster<kTotalNumKernels>
kernel_cluster(meta_list,
               [[shim_kernel_name]]_packed_string);

static [[lut_ctype]] lut[[lut_cshape]] =
[[lut_data]]
;

}; // End of anonymous namespace

namespace AOTRITON_NS::v3::[[kernel_family_name]]::autotune {

// using AOTRITON_NS::v2::[[kernel_family_name]]::[[context_class_name]];

void CURRENT_ENTRY_PUBLIC([[context_class_name]]& context, int mod_number) {
#if AOTRITON_BUILD_FOR_TUNING
    constexpr int kLutInTuningModeAlwaysReturnZero = 0;
#ifndef NDEBUG
    std::cerr << "Autotune_[[shim_kernel_name]]__A[[arch_number]]__F[[godel_number]] "
              << "kTotalNumKernels = " << kTotalNumKernels << " "
              << "_has_preferred_kernel = " << context._has_preferred_kernel << " "
              << std::endl;
    assert([[deduplicated_lut_function]](*context.params, mod_number, lut) == kLutInTuningModeAlwaysReturnZero);
#endif
    int preferred = context._has_preferred_kernel;
    int kernel_index = preferred < 0 ? kLutInTuningModeAlwaysReturnZero : preferred;
#else
    auto kernel_index = [[deduplicated_lut_function]](*context.params, mod_number, lut);
#endif
    if (kernel_index < 0 || kernel_index >= kTotalNumKernels) {
        return ;
    }
    context.kernel_on_device = kernel_cluster.get(kernel_index);
    context.pp_args_index = [[deduplicated_pp_args_function_index]];
    context.flatzip_path = FLATZIP_PATH;
    context.aks2_entry   = AKS2_ENTRY;
    context.func_name = FUNC_NAME;
    context.arch_name = ARCH_NAME;
#if AOTRITON_BUILD_FOR_TUNING
    context._total_number_of_kernels = kTotalNumKernels;
    context._preferred_kernel_psels = kernel_psels[kernel_index];
    context._preferred_kernel_copts = kernel_copts[kernel_index];
#endif
#ifndef NDEBUG
    std::cerr << __FILE__ << " kernel_index = " << int(kernel_index) << std::endl;
#endif
    const auto& perf = image_perf_list[kernel_index];
    [[perf_field_assignment]];
}

#undef CURRENT_ENTRY_PUBLIC
}

// [[human_readable_signature]]

// [[sql]]

// vim: set fileencoding=utf-8
