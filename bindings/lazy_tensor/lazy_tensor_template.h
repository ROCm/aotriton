// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AOTRITON_V3_BINDING_LAZY_TENSOR_TEMPLATE_H
#define AOTRITON_V3_BINDING_LAZY_TENSOR_TEMPLATE_H

#include <ATen/core/Tensor.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/empty.h>
#include <aotriton/util.h>

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton::lazy_tensor {

template<int kRank, bool kRequireZeros>
struct LazyTensorHelper {

  struct context {
    aotriton::TensorView<kRank> like_tensor;
    int device_index;
    at::Tensor tensor;
    std::array<uint64_t, kRank> tensor_strides;
  };

  static aotriton::TensorView<kRank> acquire(void* cookie) {
    auto ctx = (context*)cookie;
    create_tensor(ctx);
    return aotriton::TensorView<kRank>(reinterpret_cast<intptr_t>(ctx->tensor.data_ptr()),
                                       ctx->like_tensor.sizes(),
                                       ctx->tensor_strides,
                                       aotriton::DType::kFloat32);
  }

  static void dispose(void* cookie) {
    auto ctx = (context*)cookie;
    delete ctx;
  }

private:
  // FIXME: This is not optimal but easier to understand...
  static void create_tensor(context* ctx) {
    auto options = at::dtype(at::kFloat).device(at::kCUDA);
#define SZ(i) (static_cast<int64_t>(ctx->like_tensor.size(i)))
#define ST(i) (static_cast<uint64_t>(ctx->tensor.stride(i)))
    if constexpr (kRank == 4) {
      if constexpr (kRequireZeros) {
        ctx->tensor = at::zeros({ SZ(0), SZ(1), SZ(2), SZ(3) }, options);
      } else {
        ctx->tensor = at::empty({ SZ(0), SZ(1), SZ(2), SZ(3) }, options);
      }
      ctx->tensor_strides = { ST(0), ST(1), ST(2), ST(3) };
    } else if constexpr (kRank == 2) {
      if constexpr (kRequireZeros) {
        ctx->tensor = at::zeros({ SZ(0), SZ(1) }, options);
      } else {
        ctx->tensor = at::empty({ SZ(0), SZ(1) }, options);
      }
      ctx->tensor_strides = { ST(0), ST(1) };
    }
    static_assert(kRank == 4 || kRank == 2, "Unimplementd kRank in LazyTensorHelper");
#undef SZ
#undef ST
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

} // namespace pyaotriton::lazy_tensor

#endif
