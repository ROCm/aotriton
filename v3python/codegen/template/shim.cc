// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "shim.[[shim_kernel_name]].h"
#include <aotriton/util.h>
#include <tuple>

namespace AOTRITON_NS::v2::[[kernel_family_name]] {

#define CAST(x) const_cast<void*>(static_cast<const void*>(x))
typedef std::vector<void*>(*PP_FUNC)(const [[context_class_name]]& context, hipDeviceptr_t*);

namespace {
extern PP_FUNC prepare_arguments[ [[pp_func_num]] ];
}

int64_t [[context_class_name]]::godel_number() const
{
    int64_t sum = 0;
    const auto& args = *params;
[[godel_number_body]]
    return sum;
}

namespace [[shim_kernel_name]]_helpers {

hipError_t
lookup_optimal([[context_class_name]]& context, Gpu gpu) {
    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    context.kernel_on_device = nullptr;
    auto tune_func = autotune_table[arch_number][context.godel_number()];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(context, mod_number);
    if (!context.kernel_on_device)
        return hipErrorSharedObjectSymbolNotFound;
    return hipSuccess;
}

hipError_t
launch(const [[context_class_name]]& context, hipStream_t stream) {
    constexpr std::string_view triton_kernel_name { "[[triton_kernel_name]]" };
    hipDeviceptr_t global_scratch = 0;
    auto args = prepare_arguments[context.pp_args_index](context, &global_scratch);
    dim3 grid = grid_calculator(context);
#if AOTRITON_BUILD_FOR_TUNING
    return context.kernel_on_device->invoke(triton_kernel_name,
                                            contex.package_path,
                                            context.func_name,
                                            context.arch_name,
                                            grid,
                                            args,
                                            peek_kernel_image,
                                            stream);
#else
    return context.kernel_on_device->invoke(triton_kernel_name,
                                            context.package_path,
                                            context.func_name,
                                            context.arch_name,
                                            grid,
                                            args,
                                            stream);
#endif
}

std::tuple<int, int>
get_archmod_number(Gpu gpu) {
    [[get_archmod_number_body]];
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}

} // [[shim_kernel_name]]_helpers

[[list_of_pp_args_function_defs]]

namespace {
PP_FUNC prepare_arguments[ [[pp_func_num]] ] = {
  [[list_of_pp_args_function_decls]]
};
}

[[define_compiled_in_features]]

namespace autotune {

const char [[shim_kernel_name]]_packed_string[] =
[[per_kernel_packed_string]];

[[list_of_deduplicated_lut_functions]]

} // namespace autotune

namespace [[shim_kernel_name]]_helpers {

AutoTuneTableEntry
autotune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

} // [[shim_kernel_name]]_helpers

}
