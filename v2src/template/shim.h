// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <functional>
#include <string>
#include <vector>

namespace AOTRITON_NS::v2::[[kernel_family_name]] {

struct [[param_class_name]] {
    // Function related arguments
    [[func_fields]];
    // Performance related arguments for current selection
    [[perf_fields]];

    TritonKernel* kernel_on_device = nullptr;
    int pp_args_index = -1;
    std::string_view package_path;
    std::string_view func_name;
    std::string_view arch_name;
    // Note to save ELF space, this object is constructed on the fly.
    const char* _debug_kernel_name = nullptr;
#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_kernel = -1; // For C++ based autotune database generation
    int _total_number_of_kernels = -1;
    const char* _preferred_kernel_psels = nullptr;
    const char* _preferred_kernel_copts = nullptr;
#endif

    int64_t godel_number() const;
};

class [[context_class_name]] {
public:
    std::function<dim3(const [[param_class_name]]&)> grid_calculator;

    hipError_t lookup_optimal([[param_class_name]]& params, Gpu gpu);
    hipError_t launch(const [[param_class_name]]& params, hipStream_t stream);
    static std::tuple<int, int> get_archmod_number(Gpu gpu);

#if AOTRITON_BUILD_FOR_TUNING
    bool peek_kernel_image = false;
#endif
private:
    typedef void (*AutoTuneTableEntry)([[param_class_name]]& params, int mod_number);
    static AutoTuneTableEntry autotune_table[][ [[number_of_functionals]] ];
};

struct [[metadata_class_name]] {
    // Note: FEAT_CHOICES here
    [[declare_compiled_in_features]]
};

namespace autotune {

using AOTRITON_NS::v2::[[kernel_family_name]]::[[param_class_name]];

[[declare_list_of_deduplicated_lut_functions]]

[[kernel_table_entry_declares]]

}


}
