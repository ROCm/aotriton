#ifndef AOTRITON_V2_INTERNAL_AITER_HIP_COMMON_H
#define AOTRITON_V2_INTERNAL_AITER_HIP_COMMON_H

// Cross-platform diagnostic suppression macros.
#if defined(_MSC_VER)
#  define AOTRITON_COMPILER_PUSH          __pragma(warning(push))
#  define AOTRITON_COMPILER_ALLOW_NARROWING __pragma(warning(disable: 4244 4267))
#  define AOTRITON_COMPILER_POP           __pragma(warning(pop))
#elif defined(__GNUC__) || defined(__clang__)
#  define AOTRITON_COMPILER_PUSH          _Pragma("GCC diagnostic push")
#  define AOTRITON_COMPILER_ALLOW_NARROWING _Pragma("GCC diagnostic ignored \"-Wnarrowing\"")
#  define AOTRITON_COMPILER_POP           _Pragma("GCC diagnostic pop")
#else
#  define AOTRITON_COMPILER_PUSH
#  define AOTRITON_COMPILER_ALLOW_NARROWING
#  define AOTRITON_COMPILER_POP
#endif

// Cross-platform packed struct macros. MSVC does not support
// __attribute__((packed)) and requires #pragma pack instead.
#if defined(_MSC_VER)
#  define AOTRITON_PACKED_STRUCT_START __pragma(pack(push, 1))
#  define AOTRITON_PACKED_STRUCT_END   __pragma(pack(pop))
#  define AOTRITON_PACKED_ATTR
#elif defined(__GNUC__) || defined(__clang__)
#  define AOTRITON_PACKED_STRUCT_START
#  define AOTRITON_PACKED_STRUCT_END
#  define AOTRITON_PACKED_ATTR __attribute__((packed))
#else
#  define AOTRITON_PACKED_STRUCT_START
#  define AOTRITON_PACKED_STRUCT_END
#  define AOTRITON_PACKED_ATTR
#endif

#include <aotriton/config.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
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
    size_t* arg_size_ptr;
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
    Gpu gpu_ = GPU_ARCH_UNKNOWN;  // Set it to avoid duplicated query from stream_id_ in get_gpu_arch()
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
  std::string hsaco_;  // CAVEAT: the hsaco passed-in by constructor may be temporary
  mutable std::filesystem::path path_cache_;
public:
  AiterAsmKernel(const char* name, const char* hsaco);
  ~AiterAsmKernel();
  void launch_kernel(const AiterAsmKernelArgs& kargs);
  pstring_view get_package_path(hipStream_t stream, pstring_type& persistant_storage, std::string& aiter_module) const;
};

std::tuple<Gpu, std::string_view>
get_gpu_arch(const ck_tile::stream_config&);

} // namespace AOTRITON_NS::v3::aiter

#endif
