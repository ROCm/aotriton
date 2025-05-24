// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/util.h>
#include <aotriton/runtime.h>
#include <functional>
#include <string>
#include <vector>

namespace AOTRITON_NS::v3::[[family_name]] {

// Unlike KernelDescription, Operator must have its own parameter class
struct [[param_class_name]] {
    [[func_fields]];
};

struct [[context_class_name]] {
    const [[param_class_name]] *params = nullptr;
    enum BackendEnum : int32_t {
        None = -1,
        [[list_of_backend_enum]],
        Max = [[total_number_of_backends]]
    };
    static constexpr BackendEnum fallback_backend = [[fallback_backend]];
    BackendEnum backend_index = BackendEnum::None;

#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_backend = -1;
    static constexpr int _total_number_of_backends = BackendEnum::Max;
    const char* _backend_name = nullptr;
#endif

    // One more layer of dispatcher of functionals is added due to
    // 1. Individual kernel may use fewer arguments
    // 2. Metro kernel needs overall performance numbers over individual kernels.
    // 3. Even metro kernel only has one kernel, another set LUT is need to
    //    determine which metro kernel (or backend) need to be used
    int64_t godel_number() const;
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = [[number_of_functionals]];

    hipError_t lookup_optimal(Gpu gpu);
    // Unlike Triton kernel, Operator's launch need gpu argument to eventually
    // call backend's lookup_optimal
    hipError_t launch(Gpu gpu, hipStream_t stream) const;
private:
    typedef void (*OpTuneTableEntry)([[context_class_name]]& context, int mod_number);
    static OpTuneTableEntry optune_table[][ kMaxGodelNumber ];

    typedef hipError_t (*BackendLauncher)(const [[context_class_name]]& context,
                                          Gpu gpu,
                                          hipStream_t stream);
    static BackendLauncher launcher_table[ BackendEnum::Max ];
};

namespace optune {

[[declare_list_of_deduplicated_lut_functions]]

[[optune_table_entry_declares]]

}

}

// vim: set fileencoding=utf-8
