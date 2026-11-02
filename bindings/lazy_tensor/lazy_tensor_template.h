// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_BINDING_LAZY_TENSOR_TEMPLATE_H
#define AOTRITON_V3_BINDING_LAZY_TENSOR_TEMPLATE_H

#include <aotriton/util.h>
#include "torch_stable_min.h"

namespace nb = nanobind;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton::lazy_tensor {

template<int kRank, bool kRequireZeros>
struct LazyTensorHelper {

  struct context {
    aotriton::TensorView<kRank> like_tensor;
    int device_index;
    torch_stable_min::Tensor tensor;
    std::array<uint64_t, kRank> tensor_strides;
  };

  static aotriton::TensorView<kRank> acquire(aotriton::LazyTensor<kRank>* self) {
    auto ctx = (context*)self->cookie;
    create_tensor(ctx);
    return aotriton::TensorView<kRank>(reinterpret_cast<intptr_t>(ctx->tensor.data_ptr()),
                                       ctx->like_tensor.sizes(),
                                       ctx->tensor_strides,
                                       aotriton::DType::kFloat32);
  }

  static void dispose(aotriton::LazyTensor<kRank>* self) {
    auto ctx = (context*)self->cookie;
    delete ctx;
  }

  static aotriton::TensorView<kRank> acquire_eager(aotriton::LazyTensor<kRank>* self) {
    return self->eager;
  }
private:
  static void create_tensor(context* ctx) {
    std::array<int64_t, kRank> sizes;
    for (int i = 0; i < kRank; i++) {
      sizes[i] = static_cast<int64_t>(ctx->like_tensor.size(i));
    }
    ctx->tensor = torch_stable_min::empty_float32(sizes, ctx->device_index);
    if constexpr (kRequireZeros) {
      torch_stable_min::zero_(ctx->tensor);
    }
    for (int i = 0; i < kRank; i++) {
      ctx->tensor_strides[i] = static_cast<uint64_t>(ctx->tensor.stride(i));
    }
  }

}; // LazyTensorHelper

template<int kRank, bool kRequireZeros>
static auto
lazy_tensor_creator(const aotriton::TensorView<kRank>& like_tensor, int device_index) {
  using LTH= LazyTensorHelper<kRank, kRequireZeros>;
  auto cookie = new typename LTH::context;
  cookie->like_tensor = like_tensor;
  cookie->device_index = device_index;
  return aotriton::LazyTensor<kRank> {
    .cookie = cookie,
    .acquire = &LTH::acquire,
    .dispose = &LTH::dispose
  };
}

template<int kRank>
static auto
eager_lazy_tensor_creator(const aotriton::TensorView<kRank>& tensor_view) {
  using LTH= LazyTensorHelper<kRank, false>;
  return aotriton::LazyTensor<kRank> {
    .cookie = nullptr,
    .acquire = &LTH::acquire_eager,
    .dispose = nullptr,
    .eager = tensor_view
  };
}

} // namespace pyaotriton::lazy_tensor

#endif
