// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/TensorAccessor.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace {

struct dq_acc_cookie {
  TensorView<4> dq;
  int device_index;
  at::Tensor tensor;
};

TensorView<4> dq_acc_acquire(void* cookie) {
  auto ctx = (dq_acc_cookie*)cookie;
  auto options = at::dtype(at::kFloat).device(at::kCUDA);
  ctx->tensor = at::zeros(ctx->dq.sizes(), options);
}

void dq_acc_dispose(void* cookie) {
  auto ctx = (dq_acc_cookie*)cookie;
  using TA = at::TensorAccessor<at::ScalarType::Float, 4>;
  ctx->tensor.to(TA(dq.data_ptr(), &dq.sizes(), &dq.strides()));
  delete ctx;
}


}

// It is complicated to implement Lazy Tensor in pure Python
// We choose to specialize lazy_tensor for each major use
namespace pyaotriton::lazy_tensor {

  void setup_module(py::module_& m) {
    m.def("dq_acc",
          [](const aotriton::TensorView<4>& dq, int device_index) {
            auto cookie = new dq_acc_cookie;
            cookie->dq = dq;
            cookie->device_index = device_index;
            return aotriton::LazyTensor<4> {
              .cookie = cookie,
              .acquire = &dq_acc_acquire,
              .dispose = &dq_acc_dispose
            };
          });
  }


}
