// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "affine.[[affine_kernel_name]].h"
#include <aotriton/util.h>
#include <tuple>

namespace AOTRITON_NS::v3::[[kernel_family_name]] {

#if [[shared_iface]]
using AOTRITON_NS::v3::[[shared_iface_family]]::[[param_class_name]];
#endif

typedef std::vector<char>(*PP_FUNC)(const [[param_class_name]]& context, hipDeviceptr_t*);

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

hipError_t
[[context_class_name]]::lookup_optimal(Gpu gpu) {
    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    kernel_on_device = nullptr;
    // Unlike Triton's autotune_table
    // Affine kernel uses entries from "capability_table", which validate if
    // input is supported.
    auto validator = capability_table[arch_number][godel_number()];
    if (!validator)
        return hipErrorPeerAccessUnsupported;
    return validator(*this, mod_number);
}


hipError_t
[[context_class_name]]::launch(hipStream_t stream) const {
    constexpr std::string_view triton_kernel_name { "[[triton_kernel_name]]" };
    hipDeviceptr_t global_scratch = 0;
    auto args = prepare_arguments[pp_args_index](*this->params, &global_scratch);
    dim3 grid = grid_calculator();
    return kernel_on_device->invoke(triton_kernel_name,
                                    package_path,
                                    func_name,
                                    arch_name,
                                    grid,
                                    args,
                                    stream);
}

[[list_of_pp_args_function_defs]]

namespace {
PP_FUNC prepare_arguments[ [[pp_func_num]] ] = {
  [[list_of_pp_args_function_decls]]
};
}

}
