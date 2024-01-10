#pragma once
#include <string>
#include <functional>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/_internal/triton_kernel.h>

namespace aotriton::v2::[[kernel_family_name]] {

struct [[param_class_name]];

namespace autotune {
    template<int ArchNumber, int GodelNumber>
    struct Autotune_[[shim_kernel_name]] {
        void operator()([[param_class_name]]& params);
    };
}

struct [[param_class_name]] {
    // Function related arguments
    [[func_fields]];
    // Performance related arguments for current selection
    [[perf_fields]];

    int64_t godel_number() const;
    std::function<dim3(const [[param_class_name]]&)> grid_calculator;

    hipError_t lookup_optimal(GpuArch arch);
    hipError_t launch(hipStream_t stream);
    int64_t get_arch_number(GpuArch arch);

    TritonKernel* selected_kernel = nullptr;
private:
    GpuArch kernel_arch = GPU_ARCH_UNKNOWN;

    using AutoTuneTableEntry = std::function<void([[param_class_name]]& param)>;
    static AutoTuneTableEntry autotune_table[][ [[number_of_functionals]] ];
};

}
