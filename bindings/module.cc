// Copyright © 2023-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
  // Family API submodules (v2, v3, …) live in modules/<family>/bindings/ and
  // self-register via SubmoduleRegistrar; setup_registered_submodules() below
  // instantiates them. No per-family names are hardcoded here.

  namespace lazy_tensor {
    void setup_module(nb::module_& m); // Impl. goes into lazy_tensor.cc (core: common to all families)
  } // namespace lazy_tensor

  void def_stream(nb::module_& m) {
    nb::class_<aotriton::Stream>(m, "Stream").def(nb::init<>());
  }

  void def_dtype(nb::module_& m) {
#define EV(name) value(#name, aotriton::DType::name)
    nb::enum_<aotriton::DType>(m, "DType", nb::is_arithmetic())
      .EV(kUnknown)
      .EV(kFloat32)
      .EV(kFloat16)
      .EV(kBFloat16)
      .EV(kInt8)
      .EV(kInt16)
      .EV(kInt32)
      .EV(kInt64)
      .EV(kUInt8)
      .EV(kUInt16)
      .EV(kUInt32)
      .EV(kUInt64)
      .export_values();
#undef EV
  }

  void def_hipruntime(nb::module_& m);
  void def_hipmemory(nb::module_& m);

  template<int Rank>
  void def_tensorview(nb::module_& m, const std::string& name) {
    nb::class_<aotriton::TensorView<Rank>>(m, name.c_str())
      .def(nb::init<intptr_t, std::array<uint64_t, Rank>, std::array<uint64_t, Rank>, aotriton::DType>())
      .def("size", &aotriton::TensorView<Rank>::size)
      .def("stride", &aotriton::TensorView<Rank>::stride)
      .def_prop_ro("sizes", &aotriton::TensorView<Rank>::sizes)
      .def_prop_ro("strides", &aotriton::TensorView<Rank>::strides)
      .def_prop_ro("data_ptr", &aotriton::TensorView<Rank>::data_ptr)
      .def_prop_ro("dtype", &aotriton::TensorView<Rank>::dtype);
  }

  // aotriton -> pyaotriton
  void setup_module(nb::module_& m) {
    m.doc() = "AOTriton Python binding";
    def_stream(m);
    def_dtype(m);
    def_hipruntime(m);
    def_tensorview<4>(m, "T4");
    def_tensorview<2>(m, "T2");
    def_tensorview<1>(m, "T1");
    def_hipmemory(m);
    // FIXME: deduplication of T0 code
    nb::class_<aotriton::TensorView<0>>(m, "T0")
      .def(nb::init<intptr_t, aotriton::DType>())
      .def("size", &aotriton::TensorView<0>::size)
      .def("stride", &aotriton::TensorView<0>::stride)
      .def_prop_ro("sizes", &aotriton::TensorView<0>::sizes)
      .def_prop_ro("strides", &aotriton::TensorView<0>::strides)
      .def_prop_ro("data_ptr", &aotriton::TensorView<0>::data_ptr)
      .def_prop_ro("dtype", &aotriton::TensorView<0>::dtype);
    m.def("get_name_suffix",
#if AOTRITON_ENABLE_SUFFIX
#define xstr(s) str(s)
#define str(s) #s
          []() -> std::string { return xstr(AOTRITON_NAME_SUFFIX); }
#undef xstr
#undef str
#else
          []() -> std::string { return ""; }
#endif
         );
    // Family API submodules (v2, v3, …) register themselves from
    // modules/*/bindings/ and are instantiated here.
    setup_registered_submodules(m);
    nb::module_ mod_lazy_tensor = m.def_submodule("lazy_tensor", "lazy_tensor API namespace");
    lazy_tensor::setup_module(mod_lazy_tensor);
  }

} // namespace pyaotriton

NB_MODULE(pyaotriton, m) {
  pyaotriton::setup_module(m);
}
