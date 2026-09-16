#ifndef AOTRITON_V2_INTERNAL_AITER_HIP_COMMON_H
#define AOTRITON_V2_INTERNAL_AITER_HIP_COMMON_H

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

// AiterAsmKernel::launch_kernel returns void -- the vendored AITER dispatchers
// call it as a bare statement from inside void lambdas -- so a launch it REFUSES
// (no .co for this kernel in aotriton.images) has no return value to report
// itself with, and leaves the HIP error state untouched. Without this channel
// ck_tile::launch_kernel below reads the refusal as success, and a kernel that
// never ran looks exactly like one that completed.
void record_launch_error(hipError_t err);
// Reads the pending refusal and clears it; hipSuccess when there was none.
hipError_t take_launch_error();

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
    // Drop anything an earlier call left behind. Discarded on purpose: this is
    // the reset, not a check (hipError_t is [[nodiscard]]).
    static_cast<void>(take_launch_error());
    auto ran_ok = [](auto&& callable, const stream_config& s) {
      callable(s);
      // Two distinct failures: hipPeekAtLastError catches a launch that ran and
      // failed, take_launch_error catches one that never ran at all. Both are
      // evaluated -- take_launch_error must clear even when HIP already failed.
      bool hip_ok = hipPeekAtLastError() == hipSuccess;
      bool ran    = take_launch_error() == hipSuccess;
      return hip_ok && ran;
    };
    if (!((ran_ok(callables, sc)) && ...)) {
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
  // Writes path_cache_ on first call. Caller ensures thread safety; the lazy
  // callback of OnDeviceKernel::get_kernel already runs under its write lock.
  pstring_view get_package_path(hipStream_t stream, pstring_type& persistant_storage, std::string& aiter_module) const;
};

std::tuple<Gpu, std::string_view>
get_gpu_arch(const ck_tile::stream_config&);

} // namespace AOTRITON_NS::v3::aiter

#endif
