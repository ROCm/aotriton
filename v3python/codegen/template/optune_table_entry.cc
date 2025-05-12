// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../op.[[op_name]].h"
// #include <aotriton/cpp_tune.h>  // TODO: add op_tune
#include <string_view>

namespace AOTRITON_NS::v3::[[op_family_name]]::optune {

using AOTRITON_NS::v3::[[op_family_name]];

// [[human_readable_signature]]
#define CURRENT_ENTRY_PUBLIC Optune_[[op_name]]__A[[arch_number]]__F[[godel_number]]

#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(array[0]))

namespace { // Anonymous namespace

using namespace std::literals::string_view_literals;

// Without macro, our LUT will be full of things like
// [[op_param_class_name]]::MetroKernelEnum::kMetro_TritonSplit
// Literally unreadable
[[enum_macros]]

static [[lut_dtype]] lut[[lut_shape]] =
[[lut_data]]
;

}; // End of anonymous namespace

namespace AOTRITON_NS::v3::[[op_family_name]]::optune {

void CURRENT_ENTRY_PUBLIC([[op_param_class_name]]& params, int mod_number) {
#if AOTRITON_BUILD_FOR_TUNING
    int preferred_index = params._has_preferred_kernel;
    if (preferred_index != -1) {
        if (preferred_index >= params._total_number_of_metros)
            return ;
        params.metro_kernel_index = preferred_index;
        return ;
    }
#endif
    auto kernel_index = [[deduplicated_lut_function]](params, mod_number, lut);
    if (kernel_index < 0) {
        return ;
    }
}

#undef CURRENT_ENTRY_PUBLIC
#undef mangle
#undef smangle
}

// vim: set fileencoding=utf-8
