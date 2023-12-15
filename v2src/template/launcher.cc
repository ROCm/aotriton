#include "[[shim_kernel_name]].h"
#include <aotriton/_internal/arch.h>

namespace aotriton::v2::[[kernel_family_name]] {

int64_t [[param_class_name]]::godel_number() const
{
    int64_t sum = 0;
[[godel_number_body]]
    return sum;
}

hipError_t [[param_class_name]]::lookup_optimal(std::string_view arch)
{
    int64_t arch_number = aotriton::v2::get_arch_index(arch) * [[arch_godel_number]];
    kernel_image = kernel_table[arch_number + godel_number()];
}

hipError_t [[param_class_name]]::launch(hipStream_t stream)
{
    TritonKernel kernel(kernel_image);
    [[let_tensor_stride_arguments]];
    std::vector<void*> args = { [[let_kernel_arguments]] };
    dim3 grid = grid_calculator(this);
    return kernel.invoke(grid, args, stream);
}

KernelTable [[param_class_name]]::kernel_table = {
    .kernel_blobs = [[kernel_table_entries]],
};

}
