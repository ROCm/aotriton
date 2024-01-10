#include "[[shim_kernel_name]].h"
#include <aotriton/util.h>

namespace aotriton::v2::[[kernel_family_name]] {

int64_t [[param_class_name]]::godel_number() const
{
    int64_t sum = 0;
[[godel_number_body]]
    return sum;
}

hipError_t [[param_class_name]]::lookup_optimal(GpuArch arch)
{
    int64_t arch_number = get_arch_number(arch);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    selected_kernel = nullptr;
    auto tune_func = autotune_table[arch_number][godel_number()];
    tune_func(*this);
    if (!selected_kernel)
        return hipErrorSharedObjectSymbolNotFound;
    return hipSuccess;
}

hipError_t [[param_class_name]]::launch(hipStream_t stream)
{
    auto arch = getArchFromStream(stream);
    // if (!selected_kernel || kernel_arch != arch) {
    if (true) { // TODO: cache kernel lookup
        auto err = lookup_optimal(arch);
        if (err != hipSuccess) {
            return err;
        }
        kernel_arch = arch;
    }
    [[let_tensor_stride_arguments]];
    std::vector<void*> args = { [[let_kernel_arguments]] };
    dim3 grid = grid_calculator(*this);
    return selected_kernel->invoke("[[triton_kernel_name]]", grid, args, stream);
}

int64_t [[param_class_name]]::get_arch_number(GpuArch arch)
{
    [[get_arch_number_body]];
    return -1;
}

namespace autotune {
[[kernel_table_entry_declares]];
}

[[param_class_name]]::AutoTuneTableEntry
[[param_class_name]]::autotune_table[][ [[number_of_functionals]] ] = {
[[kernel_table_entries]]
};

}
