// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V2_API_UTIL_H
#define AOTRITON_V2_API_UTIL_H

#include <aotriton/config.h>
#include "dtypes.h"
#include "runtime.h"
#include <functional>
#include <stdint.h>
#include <string_view>

namespace AOTRITON_NS {

constexpr uint64_t
CAT(uint32_t high, uint32_t low) {
  uint64_t high64 = high;
  uint64_t low64 = low;
  return (high64 << 32) | low64;
}

template<typename T>
T
cdiv(T numerator, T denominator) {
  return (numerator + (denominator - 1)) / denominator;
}

// Use PCI IDs to avoid allocating numbers by ourselves
enum GpuVendor : uint32_t {
  kAMD = 0x1002,
  kNVIDIA = 0x10de,
  kINTEL = 0x8086,
};

// More bits for potential non-PCI architectures
enum GpuArch : uint64_t {
  GPU_ARCH_UNKNOWN = 0,
  GPU_ARCH_AMD_GFX90A = CAT(GpuVendor::kAMD, 0x90a),
  GPU_ARCH_AMD_GFX942 = CAT(GpuVendor::kAMD, 0x942),
  GPU_ARCH_AMD_GFX1100 = CAT(GpuVendor::kAMD, 0x1100),
  GPU_ARCH_AMD_GFX1101 = CAT(GpuVendor::kAMD, 0x1101),
  GPU_ARCH_AMD_GFX950 = CAT(GpuVendor::kAMD, 0x950),
  GPU_ARCH_AMD_GFX1201 = CAT(GpuVendor::kAMD, 0x1201),
};

template<int Rank>
class TensorView {
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

  std::array<uint64_t, Rank> strides() const {
    return strides_;
  }

  const void* data_ptr() const {
    return base_;
  }

  DType dtype() const {
    return dtype_;
  }

  static TensorView<Rank> get_null_tensor(DType dtype) {
      return TensorView<Rank>{0,
                              std::array<uint64_t, Rank>{},
                              std::array<uint64_t, Rank>{},
                              dtype};
  }
private:
  const void* base_ = nullptr;
  std::array<uint64_t, Rank> sizes_;
  std::array<uint64_t, Rank> strides_;
  DType dtype_ = kUnknown;
};

template<>
class TensorView<0> {
public:
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

  const void* data_ptr() const {
    return base_;
  }

  DType dtype() const {
    return dtype_;
  }

  static TensorView<0> get_null_tensor(DType dtype) {
      return TensorView<0>{0, dtype};
  }
private:
  const void* base_ = nullptr;
  DType dtype_ = kUnknown;
};


extern template class TensorView<1>;
extern template class TensorView<2>;
extern template class TensorView<3>;
extern template class TensorView<4>;

GpuArch getArchFromStream(hipStream_t);
bool isArchExperimentallySupported(const hipDeviceProp_t*);
int getMultiProcessorCount(hipStream_t stream);

} // namespace AOTRITON_NS

#endif
