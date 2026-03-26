// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "shim.[[shim_kernel_name]].h"
#include <aotriton/util.h>
#include <tuple>
#include <iostream>
#ifdef AOTRITON_BUILD_FOR_TUNING
#include <aotriton/cpp_tune.h>
#endif
[[includes]]

namespace AOTRITON_NS::v3::[[kernel_family_name]] {

#if [[shared_iface]]
using AOTRITON_NS::v3::[[shared_iface_family]]::[[param_class_name]];
#define KERNEL_SLOT_INDEX ([[call_options_struct]]::KernelSlot::[[triton_kernel_name]])
#endif

#define CAST(x) const_cast<void*>(static_cast<const void*>(x))
typedef std::vector<void*>(*PP_FUNC)(const [[param_class_name]]& context, const TritonAuxiliaryArguments&);

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
#if AOTRITON_BUILD_FOR_TUNING && [[shared_iface]]
    if (call_options) {
        auto& kctl = call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;

        // Check Ignore flag - skip lookup/execution if set
        if (ctrl & KernelControl::Ignore) {
            launch_condition = false;
            return hipSuccess;
        }

        // Check Skip flag - lookup but skip execution
        if (ctrl & KernelControl::Skip) {
            launch_condition = false;
        }

        // Check Manual flag - use hsaco_index if set
        if (ctrl & KernelControl::Manual) {
            _has_preferred_kernel = kctl.hsaco_index;
            peek_kernel_image = (ctrl & KernelControl::ExtractImage);
        }
    }
#endif

    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    kernel_on_device = nullptr;
    auto number = godel_number();
    if (number < 0)
        return hipErrorNotSupported;
    auto tune_func = autotune_table[arch_number][number];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(*this, mod_number);
    if (!kernel_on_device)
        return hipErrorSharedObjectSymbolNotFound;

#if AOTRITON_BUILD_FOR_TUNING && [[shared_iface]]
    if (call_options) {
        auto& kctl = call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
        uint16_t ctrl = kctl.control_bits;

        // Write total_hsacos if Probe is set
        if (ctrl & KernelControl::Probe) {
            kctl.total_hsacos = _total_number_of_kernels;
            if (ctrl & KernelControl::Manual) {
                kctl.kernel_psels = _preferred_kernel_psels;
                kctl.kernel_copts = _preferred_kernel_copts;
            }
        }
    }
#endif
    return hipSuccess;
}

hipError_t
[[context_class_name]]::launch(hipStream_t stream) const {
    if (!launch_condition)
      return hipSuccess;
    constexpr std::string_view triton_kernel_name { "[[triton_kernel_name]]" };
    TritonAuxiliaryArguments aux;
    auto args = prepare_arguments[pp_args_index](*this->params, aux);
    dim3 grid;
    if (custom_grid_calculator) {
        grid = custom_grid_calculator(*this);
    } else {
        grid = grid_calculator();
    }
#if AOTRITON_BUILD_FOR_TUNING && [[shared_iface]]
    auto ret = kernel_on_device->invoke(triton_kernel_name,
                                        package_path,
                                        func_name,
                                        arch_name,
                                        grid,
                                        args,
                                        peek_kernel_image,
                                        stream);
    if (ret != hipSuccess)
         return ret;
    auto& kctl = call_options->kernel_fine_control[KERNEL_SLOT_INDEX];
    uint16_t ctrl = kctl.control_bits;
    if (ctrl & KernelControl::Manual && ctrl & KernelControl::ExtractImage) {
        auto essentials = kernel_on_device->get_image_info_iff_decompressed();
        kctl.kernel_image = essentials.image;
        kctl.image_size = essentials.size;
    }
    return ret;
#else
    return kernel_on_device->invoke(triton_kernel_name,
                                    package_path,
                                    func_name,
                                    arch_name,
                                    grid,
                                    args,
                                    stream);
#endif
}

std::tuple<int, int>
[[context_class_name]]::get_archmod_number(Gpu gpu) {
    [[get_archmod_number_body]];
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}


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

[[context_class_name]]::AutoTuneTableEntry
[[context_class_name]]::autotune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

}

// vim: set fileencoding=utf-8
