// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../iface.[[op_name]].h"
// #include <aotriton/cpp_tune.h>  // TODO: add op_tune
#include <string_view>
#ifndef NDEBUG
#include <iostream>
#endif

#define CURRENT_ENTRY_PUBLIC Optune_[[op_name]]__A[[arch_number]]__F[[godel_number]]

#define ARRAY_SIZE(array)  (sizeof(array) / sizeof(array[0]))

namespace { // Anonymous namespace

using namespace std::literals::string_view_literals;

static [[lut_ctype]] lut[[lut_cshape]] =
[[lut_data]]
;

}; // End of anonymous namespace

namespace AOTRITON_NS::v3::[[op_family_name]]::optune {

void CURRENT_ENTRY_PUBLIC([[context_class_name]]& context, int mod_number) {
    auto backend_index = [[deduplicated_lut_function]](*context.params, mod_number, lut);
    if (backend_index < 0) {
        return ;
    }
    context.backend_index = static_cast<[[context_class_name]]::BackendEnum>(backend_index);
}

#undef CURRENT_ENTRY_PUBLIC
}

// [[human_readable_signature]]

// vim: set fileencoding=utf-8
