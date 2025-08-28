// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_INTERNAL_LAZY_TENSOR_INTERNAL_H
#define AOTRITON_V3_INTERNAL_LAZY_TENSOR_INTERNAL_H

#ifndef AOTRITON_V2_API_UTIL_H
#error "Need to include aotriton/util.h before aotriton/_internal/lazy_tensor_internal.h"
#endif

#include <aotriton/config.h>

namespace AOTRITON_NS {

template<int Rank>
struct LazyTensorInternal {
  LazyTensorInternal(const LazyTensor<Rank>& lazy)
    : lazy_(lazy) {
  }

  bool activated() const {
    return concrete_;
  }

  operator bool() const {
    return lazy_ || concrete_;
  }

#define LAZY_INIT if (!concrete_) { concrete_ = (*lazy_.acquire)(lazy_.cookie); }

  uint64_t size(int i) const {
    LAZY_INIT;
    return concrete_.size(i);
  }

  uint64_t stride(int i) const {
    LAZY_INIT;
    return concrete_.stride(i);
  }

  std::array<uint64_t, Rank> sizes() const {
    LAZY_INIT;
    return concrete_.sizes();
  }

  std::array<uint64_t, Rank> strides() const {
    LAZY_INIT;
    return concrete_.strides();
  }

  void* data_ptr() const {
    LAZY_INIT;
    return concrete_.data_ptr();
  }

  DType dtype() const {
    LAZY_INIT;
    return concrete_.dtype();
  }

  void* kparam_data_ptr() const {
    LAZY_INIT;
    return concrete_.kparam_data_ptr();
  }

  void* kparam_stride(int i) const {
    LAZY_INIT;
    return concrete_.kparam_stride(i);
  }

  TensorView<Rank>& make_concrete() {
    LAZY_INIT;
    return concrete_;
  }
#undef LAZY_INIT
protected:
  LazyTensor<Rank> lazy_;
  mutable TensorView<Rank> concrete_;
  LazyTensorInternal() { }
};

// Pretend to be an lazy tensor
template<int Rank>
struct EagerLazyTensor : public LazyTensorInternal<Rank> {
  EagerLazyTensor(const TensorView<Rank>& concrete) {
    this->concrete_ = concrete;
  }
  // Note: Don't add anything, LazyTensorInternal is not virtual and overriding
  // won't work
};

};

#endif
