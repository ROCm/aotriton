// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <functional>
#include <string>
#include <vector>

namespace AOTRITON_NS::v3::[[op_family_name]] {

struct [[op_param_class_name]] {
    // Function related arguments
    [[func_fields]];

    enum MetroKernelEnum : int32_t {
        None = -1,
        [[list_of_metro_kernel_enum]],
        Max = [[total_number_of_metro_kernels]]
    };
    MetroKernelEnum metro_kernel_index = MetroKernelEnum::None;

#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_metro = -1;
    constexpr int _total_number_of_metros = MetroKernelEnum::Max;
    const char* _metro_name = nullptr;
#endif

    // One more layer of dispatcher of functionals is added due to
    // 1. Individual kernel may use fewer arguments
    // 2. Metro kernel needs overall performance numbers over individual kernels.
    // 3. Even metro kernel only has one kernel, another set LUT is need to
    //    determine which metro kernel (or backend) need to be used
    int64_t godel_number() const;
};

class [[op_context_class_name]] {
public:
    enum KernelShimEnum : int32_t {
        [[list_of_named_kernel_shim]],
        Max = [[total_number_of_kernel_shims]]
    };
    // grid_calculators need to be defined in C++ source code
    static std::function<dim3(const [[op_param_class_name]]&)> grid_calculators[ KernelShimEnum::Max ];

    hipError_t lookup_optimal([[op_param_class_name]]& params, Gpu gpu);
    hipError_t launch(const [[op_param_class_name]]& params, hipStream_t stream);
private:
    typedef void (*OpTuneTableEntry)([[op_param_class_name]]& params, Gpu gpu);
    static OpTuneTableEntry optune_table[ [[number_of_functionals]] ];
};

namespace optune {

// using AOTRITON_NS::v3::[[op_family_name]]::[[op_param_class_name]];

[[declare_list_of_deduplicated_lut_functions]]

[[kernel_table_entry_declares]]

}

}
