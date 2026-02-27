// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
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
    const char* check_inputs_are_supported();
#if [[has_cookie_object]]
    [[cookie_class]] cookie;
#endif

    hipError_t lookup_optimal(Gpu gpu);
    hipError_t launch(hipStream_t stream) const;

    static std::tuple<int, int> get_archmod_number(Gpu gpu);
};

}

// vim: set fileencoding=utf-8

