#ifndef AOTRITON_V2_INTERNAL_AITER_HIP_COMMON_H
#define AOTRITON_V2_INTERNAL_AITER_HIP_COMMON_H

#include <aotriton/config.h>
#include <aotriton/runtime.h>
#include "on_device_kernel.h"

//
// AITER/CK Compatitility code
// Must wrap with AOTRITON_NS to avoid naming conflicts
//
namespace AOTRITON_NS::v3::aiter {

struct p3
{
    unsigned int _p0;
    unsigned int _p1;
    unsigned int _p2;
};
struct p2
{
    unsigned int _p0;
    unsigned int _p1;
};
struct p1
{
    unsigned int _p0;
};
struct AiterAsmKernelArgs
{
    void* args_ptr;
    void* arg_size_ptr;
    int gdx;
    int gdy;
    int gdz;
    int bdx;
    int bdy;
    int bdz;
    const hipStream_t stream;
};

namespace ck_tile {
  using index_t = int32_t;

  template <typename T>
  struct log2e;

  template <>
  struct log2e<double>
  {
      static constexpr double value = 1.44269504088896340736;
  };

  template <>
  struct log2e<float>
  {
      static constexpr float value = float(log2e<double>::value);
  };

  template <typename T = double>
  constexpr T log2e_v = log2e<T>::value;

  template <typename T = double>
  constexpr T log2e_rcp_v = 1. / log2e<T>::value;

  struct stream_config {
    hipStream_t stream_id_;
  };
  // Simplified from include/ck_tile/host/kernel_launch.hpp
  template <typename... Callables>
  float launch_kernel(const stream_config& sc, Callables&&... callables)
  {
    if (!((static_cast<void>(callables(sc)), hipPeekAtLastError() == hipSuccess) && ...)) {
      return -1.0;
    }
    return 0;
  }
} // ck_tile

class AiterAsmKernel : public OnDeviceKernel {
private:
  const char* mangled_kernel_function_name_;
  const char* hsaco_;
  mutable std::filesystem::path path_cache_;
public:
  AiterAsmKernel(const char* name, const char* hsaco);
  ~AiterAsmKernel();
  void launch_kernel(const AiterAsmKernelArgs& kargs);
  std::string_view get_package_path(hipStream_t stream, std::string& persistant_storage) const;
};

std::tuple<uint64_t, std::string_view>
get_gpu_arch(hipStream_t);

} // namespace AOTRITON_NS::v3::aiter

#endif
