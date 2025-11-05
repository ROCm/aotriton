// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_UTIL_H
#define AOTRITON_V2_API_UTIL_H

#include <aotriton/config.h>
#include "dtypes.h"
#include "runtime.h"
#include <functional>
#include <cstdint>
#include <string_view>
#include <array>

namespace AOTRITON_NS {

constexpr uint64_t
CAT64(uint32_t high, uint32_t low) {
  uint64_t high64 = high;
  uint64_t low64 = low;
  return (high64 << 32) | low64;
}

constexpr uint32_t
CAT32(uint16_t high, uint16_t low) {
  uint32_t high32 = high;
  uint32_t low32 = low;
  return (high32 << 16) | low32;
}

constexpr uint64_t
TRICAT(uint16_t high, uint16_t mid, uint16_t low) {
  uint64_t high64 = high;
  uint64_t mid64 = mid;
  uint64_t low64 = low;
  return (high64 << 32) | (mid64 << 16) | low64;
}

template<typename T>
T AOTRITON_API
cdiv(T numerator, T denominator) {
  return (numerator + (denominator - 1)) / denominator;
}

// Use PCI IDs to avoid allocating numbers by ourselves
enum AOTRITON_API GpuVendor : uint16_t {
  kAMD = 0x1002,
  kNVIDIA = 0x10de,
  kINTEL = 0x8086,
};

// More bits for potential non-PCI architectures
enum AOTRITON_API Gpu : uint64_t {
  GPU_ARCH_UNKNOWN = 0,
  GPU_AMD_ARCH_GFX908_MOD0  = TRICAT(GpuVendor::kAMD,  0x908, 0),
  GPU_AMD_ARCH_GFX90A_MOD0  = TRICAT(GpuVendor::kAMD,  0x90a, 0),
  GPU_AMD_ARCH_GFX942_MOD0  = TRICAT(GpuVendor::kAMD,  0x942, 0),
  GPU_AMD_ARCH_GFX942_MOD1  = TRICAT(GpuVendor::kAMD,  0x942, 1),
  GPU_AMD_ARCH_GFX942_MOD2  = TRICAT(GpuVendor::kAMD,  0x942, 2),
  GPU_AMD_ARCH_GFX1100_MOD0 = TRICAT(GpuVendor::kAMD, 0x1100, 0),
  GPU_AMD_ARCH_GFX1101_MOD0 = TRICAT(GpuVendor::kAMD, 0x1101, 0),
  GPU_AMD_ARCH_GFX1102_MOD0 = TRICAT(GpuVendor::kAMD, 0x1102, 0),
  GPU_AMD_ARCH_GFX1151_MOD0 = TRICAT(GpuVendor::kAMD, 0x1151, 0),
  GPU_AMD_ARCH_GFX1150_MOD0 = TRICAT(GpuVendor::kAMD, 0x1150, 0),
  GPU_AMD_ARCH_GFX950_MOD0  = TRICAT(GpuVendor::kAMD,  0x950, 0),
  GPU_AMD_ARCH_GFX1201_MOD0 = TRICAT(GpuVendor::kAMD, 0x1201, 0),
  GPU_AMD_ARCH_GFX1200_MOD0 = TRICAT(GpuVendor::kAMD, 0x1200, 0),
  GPU_AMD_ARCH_GFX1250_MOD0 = TRICAT(GpuVendor::kAMD, 0x1250, 0),
};

inline uint32_t Gpu2VendorArch(uint64_t gpu) {
  return (gpu & 0xFFFF'FFFF'0000) >> 16;
}

template<int Rank>
class AOTRITON_API TensorView {
public:
  TensorView() {
  }

  TensorView(intptr_t base, std::array<uint64_t, Rank> sizes, std::array<uint64_t, Rank> strides, DType dtype)
    : base_(reinterpret_cast<void*>(base))
    , sizes_(sizes)
    , strides_(strides)
    , dtype_(dtype) {
  }

  // Use to enclose aten::Tensor
  template<typename Tensor>
  TensorView(const Tensor& tensor, std::function<DType(const Tensor&)> dtype_extractor) {
    base_ = tensor.data_ptr();
    for (int i = 0; i < Rank; i++) {
      sizes_ = tensor.size(i);
      strides_ = tensor.stride(i);
    }
    dtype_ = dtype_extractor(tensor);
  }

  operator bool() const {
    return base_ != nullptr;
  }

  uint64_t size(int i) const {
    return sizes_[i];
  }

  uint64_t stride(int i) const {
    return strides_[i];
  }

  std::array<uint64_t, Rank> sizes() const {
    return sizes_;
  }

  const uint64_t* size_ptr() const {
    return sizes_.data();
  }

  std::array<uint64_t, Rank> strides() const {
    return strides_;
  }

  const uint64_t* stride_ptr() const {
    return strides_.data();
  }

  void* data_ptr() const {
    return base_;
  }

  DType dtype() const {
    return dtype_;
  }

  // For hipModuleLaunchKernel's kernelParams
  void* kparam_data_ptr() const {
    return const_cast<void*>(static_cast<const void*>(&base_));
  }

  // For hipModuleLaunchKernel's kernelParams
  void* kparam_stride(int i) const {
    return const_cast<void*>(static_cast<const void*>(&strides_[i]));
  }

  static TensorView<Rank> get_null_tensor(DType dtype) {
      return TensorView<Rank>{0,
                              std::array<uint64_t, Rank>{},
                              std::array<uint64_t, Rank>{},
                              dtype};
  }
private:
  void* base_ = nullptr;
  std::array<uint64_t, Rank> sizes_;
  std::array<uint64_t, Rank> strides_;
  DType dtype_ = kUnknown;
};

template<>
class AOTRITON_API TensorView<0> {
public:
  TensorView() {
  }

  TensorView(intptr_t base, DType dtype)
    : base_(reinterpret_cast<void*>(base))
    , dtype_(dtype) {
  }

  operator bool() const {
    return base_ != nullptr;
  }

  uint64_t size(int i) const {
    return i == 0 ? 1 : 0;
  }

  uint64_t stride(int i) const {
    return i == 0 ? 1 : 0;
  }

  std::array<uint64_t, 0> sizes() const {
    return {};
  }

  std::array<uint64_t, 0> strides() const {
    return {};
  }

  void* data_ptr() const {
    return base_;
  }

  DType dtype() const {
    return dtype_;
  }

  static TensorView<0> get_null_tensor(DType dtype) {
      return TensorView<0>{0, dtype};
  }

  // For hipModuleLaunchKernel's kernelParams
  void* kparam_data_ptr() const {
    return const_cast<void*>(static_cast<const void*>(&base_));
  }
private:
  void* base_ = nullptr;
  DType dtype_ = kUnknown;
};

#ifndef aotriton_v2_EXPORTS
extern template class TensorView<1>;
extern template class TensorView<2>;
extern template class TensorView<3>;
extern template class TensorView<4>;
#endif // aotriton_v2_EXPORTS

// Lazy allocated Tensors
// For tensors that are only needed by certain backend of arguments
//
// Intentionally not using std::function to avoid potential ABI change in
// libstdc++/libc++
template<int Rank>
struct LazyTensor {
  void* cookie = nullptr;
  TensorView<Rank> (*acquire)(void* cookie) = nullptr;
  // Note for user: Remeber put necessary information to dispose this tensor to
  //                "cookie" object in acquire.
  void  (*dispose)(void* cookie) = nullptr;

  operator bool() const {
    return cookie != nullptr || acquire != nullptr || dispose != nullptr;
  }

  // FIXME: This design is prone to memory leaks.
  void free() {
    if (dispose && cookie) {
      (*dispose)(cookie);
      cookie = nullptr;
    }
  }
};

Gpu AOTRITON_API getGpuFromStream(hipStream_t);
bool AOTRITON_API isArchExperimentallySupported(hipStream_t);
int AOTRITON_API getMultiProcessorCount(hipStream_t stream);

} // namespace AOTRITON_NS

#endif
