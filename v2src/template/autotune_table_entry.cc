#define INCBIN_PREFIX g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE

#define mangle(x) g_aotriton_FAMILY_[[kernel_family_name]]_KERNEL_[[shim_kernel_name]]_GPU_[[gpu]]_ ## x ## _data

#include <incbin.h>
#include <aotriton/_internal/triton_kernel.h>
#include "../[[shim_kernel_name]].h"

[[incbin_kernel_images]];

namespace aotriton::v2::[[kernel_family_name]]::autotune {

// [[human_readable_signature]]
template<>
struct Autotune_[[shim_kernel_name]] <[[arch_number]], [[godel_number]]> {
    struct PerfFields {
        [[perf_fields]];
    };
    static TritonKernel image_list [];
    static PerfFields image_perf_list [];
    static [[lut_dtype]] lut[[lut_shape]];

    void operator()([[param_class_name]]& params) {
        [[binning_autotune_keys]]
        auto kernel_index = lut[[binned_indices]];
        params.selected_kernel = &image_list[kernel_index];
        const auto& perf = image_perf_list[kernel_index];
        [[perf_field_assignment]];
    }
};

TritonKernel Autotune_[[shim_kernel_name]] <[[arch_number]], [[godel_number]]>::image_list [] = {
        [[kernel_image_objects]]
};

Autotune_[[shim_kernel_name]] <[[arch_number]], [[godel_number]]>::PerfFields
Autotune_[[shim_kernel_name]] <[[arch_number]], [[godel_number]]>::image_perf_list [] = {
        [[kernel_image_perfs]]
    };

[[lut_dtype]] Autotune_[[shim_kernel_name]] <[[arch_number]], [[godel_number]]>::lut[[lut_shape]] = [[lut_data]];

}
