// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <pybind11/pybind11.h>
#include "lazy_tensor_template.h"

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton::lazy_tensor {
  void setup_dq_acc(py::module_& m);

  template<int Rank>
  void def_lazytensor(py::module_& m, const std::string& name) {
    // All fields are hidden
    py::class_<aotriton::LazyTensor<Rank>>(m, name.c_str());
  }

  void setup_module(py::module_& m) {
    def_lazytensor<4>(m, "LT4");
    def_lazytensor<2>(m, "LT2");
    m.def("dq_acc", &lazy_tensor_creator<4, true>);
    m.def("delta", &lazy_tensor_creator<2, false>);
  }
}
