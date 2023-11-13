#define INCBIN_PREFIX g_oort_kernel_for_shim_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin.h>
#include "oort_kernel.h"
#include "{shim_kernel_name}.h"

INCBIN({incbin_symbol_name}, "{hsaco_kernel_path}");

namespace oort {{

template<> hipError_t
{shim_kernel_name}<{shim_kernel_specialization}
                  >::operator()(dim3 grid, dim3 block, {shim_arguments}, hipStream_t stream) {{
  static oort::OortKernel kernel("{hsaco_kernel_name}",
                                 g_oort_kernel_for_shim_{incbin_symbol_name}_data,
                                 {shared_memory_size});
  std::vector<void*> args = {{ {casted_shim_parameters} }};
  return kernel.invoke(grid, block, args, stream);
}}

}}