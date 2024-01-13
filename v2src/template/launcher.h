#pragma once
#include <string>
#include <functional>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/_internal/triton_kernel.h>

namespace aotriton::v2::[[kernel_family_name]] {

struct [[param_class_name]] {
    // Function related arguments
    [[func_fields]];
    // Performance related arguments for current selection
    [[perf_fields]];

    TritonKernel* selected_kernel = nullptr;

    int64_t godel_number() const;
};

class [[context_class_name]] {
public:
    std::function<dim3(const [[param_class_name]]&)> grid_calculator;

    hipError_t lookup_optimal([[param_class_name]]& params, GpuArch arch);
    hipError_t launch(const [[param_class_name]]& params, hipStream_t stream);
    int64_t get_arch_number(GpuArch arch);

private:
    GpuArch kernel_arch = GPU_ARCH_UNKNOWN;

    using AutoTuneTableEntry = std::function<void([[param_class_name]]& params)>;
    static AutoTuneTableEntry autotune_table[][ [[number_of_functionals]] ];
};

namespace autotune {

using aotriton::v2::[[kernel_family_name]]::[[param_class_name]];

[[kernel_table_entry_declares]]

}


}
