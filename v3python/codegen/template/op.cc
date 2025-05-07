// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "op.[[op_name]].h"
#include <aotriton/util.h>
#include <tuple>
[[include_shim_kernel_headers]]

namespace AOTRITON_NS::v3::[[op_family_name]] {

// namespace shim = AOTRITON_NS::v2::[[kernel_family_name]];

typedef bool(*REQUIREMENT_FUNC)(const [[op_param_class_name]]& params);
typedef hipError_t (*METRO_LAUNCH_FUNC)(const [[op_param_class_name]]& params,
                                        const [[op_context_class_name]]& context,
                                        hipStream_t);

namespace {
constexpr auto FALLBACK_METRO_KERNEL = MetroKernelEnum::[[fallback]];
extern REQUIREMENT_FUNC requirement_functions[ [[total_number_of_metro_kernels]] ];
extern METRO_LAUNCH_FUNC launch_functions[ [[total_number_of_metro_kernels]] ];
// We store the string in the database to avoid re-ordering the metro kernels accidentally.
const char* kMetroKernelNames [] = {
  [[list_of_named_kernel_namestrings]]
};

}

int64_t [[op_param_class_name]]::godel_number() const
{
    int64_t sum = 0;
[[godel_number_body]]
    return sum;
}

hipError_t
[[op_context_class_name]]::lookup_optimal([[op_param_class_name]]& params, Gpu gpu) {
    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    params.metro_kernel_index = MetroKernelEnum::None;
    auto tune_func = optune_table[arch_number][params.godel_number()];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(params, mod_number);
    if (params.metro_kernel_index >= 0 &&
        requirement_functions[params.metro_kernel_index] &&
        !requirement_functions[params.metro_kernel_index](params, gpu))
        params.metro_kernel_index = FALLBACK_METRO_KERNEL;
    if (params.metro_kernel_index < 0)
        params.metro_kernel_index = FALLBACK_METRO_KERNEL;
#if AOTRITON_BUILD_FOR_TUNING
    if (params.metro_kernel_index < MetroKernelEnum::Max)
        params._metro_name = kMetroKernelNames[params.metro_kernel_index];
    else
        params._metro_name = nullptr;
#endif
    return hipSuccess;
}

hipError_t
[[op_context_class_name]]::launch(const [[op_param_class_name]]& params, hipStream_t stream) {
    if (params.metro_kernel_index < 0) {
        return hipErrorPriorLaunchFailure;
    }
    return launch_functions[params.metro_kernel_index](params, stream);
}

namespace {
[[list_of_requirement_functions_defs]]

REQUIREMENT_FUNC requirement_functions[ [[total_number_of_metro_kernels]] ] {
    [[list_of_requirement_functions_decls]]
};

[[list_of_metro_launch_functions_defs]]
METRO_LAUNCH_FUNC launch_functions[ [[total_number_of_metro_kernels]] ];
    [[metro_launch_entries]]
}

namespace metrotune {

[[list_of_deduplicated_lut_functions]]

} // namespace autotune

[[op_context_class_name]]::OpTuneTableEntry
[[op_context_class_name]]::optune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

}
