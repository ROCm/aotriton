// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/cpp_tune.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <string>
#include "submodule_registry.h"

namespace nb = nanobind;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton {
  namespace v2 {
    namespace flash {
      void setup_module(nb::module_& m) {
        m.def("check_gpu", &aotriton::v2::flash::check_gpu, nb::arg("stream"));
        m.def("debug_simulate_encoded_softmax",
              &aotriton::v2::flash::debug_simulate_encoded_softmax,
              "Flash Attention Debugging Function to get raw RNG numbers used in dropout",
              nb::call_guard<nb::gil_scoped_release>(),
              nb::arg("r"),
              nb::arg("dropout_p"),
              nb::arg("philox_seed_ptr"),
              nb::arg("philox_offset1"),
              nb::arg("philox_offset2"),
              nb::arg("stream") = nullptr);
      }
    } // namespace flash

    void setup_module(nb::module_& m) {
      nb::module_ mod_flash = m.def_submodule("flash", "Flash Attention API");
      flash::setup_module(mod_flash);
    }
  } // namespace v2

  namespace {
    // Self-register the v2 API submodule (see submodule_registry.h).
    const SubmoduleRegistrar _v2_registrar("v2", "v2 API namespace", &v2::setup_module);
  } // namespace
} // namespace pyaotriton
