// Copyright © 2023-2025 Advanced Micro Devices, Inc.
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
// There is no common operator for [[shim_kernel_name]].
struct [[param_class_name]] {
    [[func_fields]];
};
#endif

struct [[context_class_name]] {
    const [[param_class_name]] *params = nullptr;
    // Performance related arguments for current selection
    [[perf_fields]];

    TritonKernel* kernel_on_device = nullptr;
    int pp_args_index = -1;
    pstring_view package_path;
    std::string_view func_name;
    std::string_view arch_name;
    // Note to save ELF space, this object is constructed on the fly.
    const char* _debug_kernel_name = nullptr;
#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_kernel = -1; // For C++ based autotune database generation
    int _total_number_of_kernels = -1;
    const char* _preferred_kernel_psels = nullptr;
    const char* _preferred_kernel_copts = nullptr;
    bool peek_kernel_image = false;
#endif

    hipError_t lookup_optimal(Gpu gpu);
    hipError_t launch(hipStream_t stream) const;

    dim3 grid_calculator() const;
    std::function<dim3(const [[context_class_name]]&)> custom_grid_calculator;

    int64_t godel_number() const;
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = [[number_of_functionals]];

    typedef void (*AutoTuneTableEntry)([[context_class_name]]& context, int mod_number);
    static AutoTuneTableEntry autotune_table[][ kMaxGodelNumber ];
};

struct [[metadata_class_name]] {
    // Note: FEAT_CHOICES here
    [[declare_compiled_in_features]]
};

namespace autotune {

extern const char [[shim_kernel_name]]_packed_string[];

[[declare_list_of_deduplicated_lut_functions]]

[[kernel_table_entry_declares]]

}


}

// vim: set fileencoding=utf-8
