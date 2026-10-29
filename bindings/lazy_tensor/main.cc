// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <nanobind/nanobind.h>
#include "lazy_tensor_template.h"

namespace nb = nanobind;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton::lazy_tensor {
  void setup_dq_acc(nb::module_& m);

  template<int Rank>
  void def_lazytensor(nb::module_& m, const std::string& name) {
    // All fields are hidden. acquire/free exist for tests: acquire()
    // materializes the tensor as the library would on first use.
    nb::class_<aotriton::LazyTensor<Rank>>(m, name.c_str())
      .def("acquire", [](aotriton::LazyTensor<Rank>& self) -> aotriton::TensorView<Rank> {
        if (self.eager) {
          return self.eager;
        }
        return (*self.acquire)(&self);
      })
      .def("free", &aotriton::LazyTensor<Rank>::free);
  }

  void setup_module(nb::module_& m) {
    def_lazytensor<4>(m, "LT4");
    def_lazytensor<2>(m, "LT2");
    m.def("dq_acc", &lazy_tensor_creator<4, true>);
    m.def("delta", &lazy_tensor_creator<2, false>);
    m.def("eager_delta", &eager_lazy_tensor_creator<2>);
    // TODO(next-cycle): rename to eager_dq_acc — this API is not specific to null tensors;
    //   pass a T4 with data_ptr=0 only when you are confident dq_acc will not be used.
    m.def("eager_null_dq_acc", &eager_lazy_tensor_creator<4>);
  }
}
