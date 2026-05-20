// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/cpp_tune.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/gil.h>
#include <string>

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton {
  namespace v2 {
    namespace flash {
      void setup_module(py::module_& m) {
        m.def("check_gpu", &aotriton::v2::flash::check_gpu, py::arg("stream"));
        m.def("debug_simulate_encoded_softmax",
              &aotriton::v2::flash::debug_simulate_encoded_softmax,
              "Flash Attention Debugging Function to get raw RNG numbers used in dropout",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("r"),
              py::arg("dropout_p"),
              py::arg("philox_seed_ptr"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("stream") = nullptr);
      }
    } // namespace flash

    void setup_module(py::module_& m) {
      py::module_ mod_flash = m.def_submodule("flash", "Flash Attention API");
      flash::setup_module(mod_flash);
    }
  } // namespace v2
} // namespace pyaotriton
