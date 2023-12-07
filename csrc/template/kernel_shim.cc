#define INCBIN_PREFIX g_aotriton_kernel_for_shim_
#define INCBIN_STYLE INCBIN_STYLE_SNAKE
#include <incbin.h>
#include "aotriton_kernel.h"
#include "{shim_kernel_name}.h"

INCBIN({incbin_symbol_name}, "{hsaco_kernel_path}");

namespace aotriton::v1 {{

template<> hipError_t
{shim_kernel_name}<{shim_kernel_specialization}
                  >::operator()(dim3 grid, {shim_arguments}, hipStream_t stream) {{
  dim3 block {{ {num_warps} * {warp_size}, 1, 1 }};
  static aotriton::v1::AOTritonKernel kernel("{hsaco_kernel_name}",
                                      g_aotriton_kernel_for_shim_{incbin_symbol_name}_data,
                                      block,
                                      {shared_memory_size});
  std::vector<void*> args = {{ {casted_shim_parameters} }};
  return kernel.invoke(grid, args, stream);
}}

}}
