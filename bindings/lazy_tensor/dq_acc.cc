// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <ATen/ops/zeros.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace {

static constexpr int kRank = 4;

struct dq_acc_cookie {
  aotriton::TensorView<kRank> dq;
  int device_index;
  at::Tensor tensor;
  std::array<uint64_t, kRank> tensor_strides;
};

aotriton::TensorView<kRank> dq_acc_acquire(void* cookie) {
  auto ctx = (dq_acc_cookie*)cookie;
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
#define SZ(i)   (static_cast<int64_t>(ctx->dq.size(i)))
  ctx->tensor = at::zeros( {SZ(0), SZ(1), SZ(2), SZ(3)}, options);
#undef SZ
#define ST(i)   (static_cast<uint64_t>(ctx->tensor.stride(i)))
  ctx->tensor_strides = {ST(0), ST(1), ST(2), ST(3)};
#undef ST
  return aotriton::TensorView<kRank>(reinterpret_cast<intptr_t>(ctx->tensor.data_ptr()),
                                     ctx->dq.sizes(),
                                     ctx->tensor_strides,
                                     aotriton::DType::kFloat32);
}

void dq_acc_dispose(void* cookie) {
  auto ctx = (dq_acc_cookie*)cookie;
  // Difficult to implement dq_acc.to(dq) in C++
  // Move this to dedicated Triton kernel.
  delete ctx;
}


}

// It is complicated to implement Lazy Tensor in pure Python
// We choose to specialize lazy_tensor for each major use
namespace pyaotriton::lazy_tensor {

  void setup_dq_acc(py::module_& m) {
    m.def("dq_acc",
          [](const aotriton::TensorView<kRank>& dq, int device_index) {
            auto cookie = new dq_acc_cookie;
            cookie->dq = dq;
            cookie->device_index = device_index;
            return aotriton::LazyTensor<kRank> {
              .cookie = cookie,
              .acquire = &dq_acc_acquire,
              .dispose = &dq_acc_dispose
            };
          });
  }

}
