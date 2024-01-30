#define INCBIN_PREFIX g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE

#define mangle(x) g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_ ## x ## _data
#define smangle(x) g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_ ## x ## _size

#include <incbin.h>
#include <aotriton/_internal/triton_kernel.h>
#include "../shim.[[shim_kernel_name]].h"
#include <iostream>

// [[human_readable_signature]]
#define CURRENT_ENTRY_PUBLIC Autotune_[[shim_kernel_name]]__A[[arch_number]]__F[[godel_number]]

[[incbin_kernel_images]];

namespace { // Anonymous namespace

struct PerfFields {
    [[perf_fields]];
};

PerfFields image_perf_list [] = {
    [[kernel_image_perfs]]
};

aotriton::TritonKernel image_list [] = {
    [[kernel_image_objects]]
};

[[lut_dtype]] lut[[lut_shape]] = [[lut_data]];

}; // End of anonymous namespace

namespace aotriton::v2::[[kernel_family_name]]::autotune {

// using aotriton::v2::[[kernel_family_name]]::[[param_class_name]];

void CURRENT_ENTRY_PUBLIC::operator()([[param_class_name]]& params) {
    [[binning_autotune_keys]]
    auto kernel_index = lut[[binned_indices]];
    std::cerr << __FILE__ << " kernel_index = " << int(kernel_index) << std::endl;
    params.selected_kernel = &image_list[kernel_index];
    const auto& perf = image_perf_list[kernel_index];
    [[perf_field_assignment]];
}

#undef CURRENT_ENTRY_PUBLIC
#undef mangle
#undef smangle
}
