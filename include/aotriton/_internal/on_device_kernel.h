// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_API_BINARY_KERNEL_H
#define AOTRITON_V3_API_BINARY_KERNEL_H

#include <atomic>
#include <memory>
#include <filesystem>
#include <string_view>
#include <shared_mutex>
#include <unordered_map>
#include <functional>
#include <tuple>
#include <aotriton/config.h>
#include <aotriton/runtime.h>

using pstring_type = std::filesystem::path::string_type;
using pstring_view = std::basic_string_view<std::filesystem::path::value_type>;

#if defined(_WIN32)
// On Windows pstring_view is wstring_view, which std::format cannot emit as
// UTF-8 text.  This specialisation converts via filesystem::path::u8string()
// so AOTRITON_LOG("{}", flatzip_path) works without manual conversion at every
// call site.  On Linux pstring_view == string_view and needs no specialisation.
#include <format>
template <>
struct std::formatter<pstring_view> : std::formatter<std::string_view> {
  auto format(pstring_view sv, std::format_context& ctx) const {
    auto u8 = std::filesystem::path(sv).u8string();
    return std::formatter<std::string_view>::format(
        std::string_view(reinterpret_cast<const char*>(u8.data()), u8.size()), ctx);
  }
};
#endif

namespace AOTRITON_NS {

class PackedKernel;

class OnDeviceKernel {
public:
  // TODO: Rename this to InMemoryKernel
  struct Essentials {
    const void* image = nullptr;
    size_t size = 0;
    int shared_memory_size = 0;
    dim3 block { 0, 0, 0 };  // For Kernel who has compile-time determined block size
  };
  struct OnDiskKernelInfo {
    pstring_view     flatzip_path;  // path to .zip relative to aotriton.images/
    std::string_view aks2_entry;    // ZIP entry name = unified_signature
    std::string_view stem_name;     // HSACO entry name inside AKS2 (;;#F/P/CO/arch)
    std::string_view function_name; // HIP kernel symbol name
  };

  OnDeviceKernel() {
  }
  ~OnDeviceKernel();

  // TODO: Make it const and add mutable to members
  std::tuple<hipFunction_t, const Essentials&> get_kernel(int device_id,
                                                          std::function<OnDiskKernelInfo()> lazy);
  void clear_device_kernel();
  void clear_decompressed_image();
#if AOTRITON_BUILD_FOR_TUNING
  // Will not work unless invoke is called at least once, i.e., If-and-only-iF decompressed
  Essentials get_image_info_iff_decompressed() const;
#endif
private:
  std::atomic<bool> kernel_loaded_ = false;

  hipFunction_t cfind_function(int device_id) const;
  // AKS2 kernel (-> In-Memory kernel image) -> hipFunction_t on certain given device
  std::tuple<hipFunction_t, hipError_t> load_for_device(int device_id,
                                                        const OnDiskKernelInfo& info);
  struct DeviceFunction {
    DeviceFunction(int device_id_, hipModule_t mod_, hipFunction_t func_);
    ~DeviceFunction();
    int device_id = -1;
    hipModule_t mod = nullptr;
    hipFunction_t func = nullptr;
  };
  std::unordered_map<int, DeviceFunction> funcache_;
  std::shared_mutex funcache_mutex_;

  // AKS2 kernel -> In-Memory kernel image
  Essentials essentials_;
  void decompress_kernel(const OnDiskKernelInfo& info);
  std::shared_ptr<PackedKernel> packed_kernel_ = nullptr;
  std::shared_mutex packedkernel_mutex_;
};

}

#endif // AOTRITON_V3_API_BINARY_KERNEL_H
