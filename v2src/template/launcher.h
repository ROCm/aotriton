#pragma once
#include <string>
#include <aotriton/_internal/kernel_table.h>

namespace aotriton::v2::[[kernel_family_name]] {

struct [[param_class_name]] {
    void* kernel_image = nullptr;
    // Function related arguments
    [[func_fields]];
    // Performance related arguments
    [[perf_fields]];

    int64_t godel_number() const;
    std::function<dim3(const [[param_class_name]]&)> grid_calculator;

    hipError_t lookup_optimal(std::string_view arch);
    hipError_t launch(hipStream_t stream);
private:
    static KernelTable kernel_table;
};

}
