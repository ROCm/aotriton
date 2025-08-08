// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <functional>
#include <string>
#include <vector>
[[includes]]

#if [[shared_iface]]
namespace AOTRITON_NS::v3::[[shared_iface_family]] {
    struct [[param_class_name]];
}
#endif

namespace AOTRITON_NS::v3::[[kernel_family_name]] {

#if [[shared_iface]]
using AOTRITON_NS::v3::[[shared_iface_family]]::[[param_class_name]];
#else
// The parameter class must be defined here when
// There is no common operator for [[affine_kernel_name]].
struct [[param_class_name]] {
    [[func_fields]];
};
#endif

struct [[context_class_name]] {
    const [[param_class_name]] *params = nullptr;
    struct {
        [[residual_func_fields]];
    } residual_args;
    struct {
        [[csv_perf_fields]]
    } perf_args;
    const char* check_inputs_are_supported();
    void calculate_residual_func_fields();

    // Re-use TritonKernel class
    TritonKernel* kernel_on_device = nullptr;

    // Kernel arguments
    union DirectKernelArguments {
        [[union_of_possible_structs]]
    };
    typedef std::tuple<dim3, dim3>([[context_class_name]]::*PP_FUNC)(DirectKernelArguments&) const;
    // These functions will be defined in
    // v3src/<family>/affine_<kernel_name>.cc
    [[pp_func_decls]];
    PP_FUNC selected_pp_args;
    size_t sizeof_selected_args;

    // Kernel locator
    std::string_view affine_kernel_function_name;
    pstring_view package_path;
    std::string_view arch_name;
    // Note to save ELF space, this object is constructed on the fly.
    const char* _debug_kernel_name = nullptr;
#if AOTRITON_BUILD_FOR_TUNING
#endif

    hipError_t lookup_optimal(Gpu gpu);
    hipError_t launch(hipStream_t stream) const;

    std::function<dim3(const [[context_class_name]]&)> custom_grid_calculator;
    dim3 grid;

    int64_t godel_number() const;
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = [[number_of_functionals_with_residuals]];

    typedef hipError_t (*CapabilityTableEntry)([[context_class_name]]& context, int mod_number);
    static CapabilityTableEntry capability_table[][ kMaxGodelNumber ];
};

}

// vim: set fileencoding=utf-8

