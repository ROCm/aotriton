// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/cpp_tune.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/gil.h>
#include "submodule_registry.h"

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton::v3 {
  namespace flash {
      using aotriton::v3::flash::attn_fwd_params;
      using aotriton::v3::flash::attn_bwd_params;
      using aotriton::v3::flash::attn_options;
      void setup_module(py::module_& m) {
        auto attn_options_class = py::class_<attn_options>(m, "attn_options")
          .def(py::init<>())
          .def_readwrite("force_backend_index", &attn_options::force_backend_index)
          .def_readwrite("deterministic", &attn_options::deterministic)
#if AOTRITON_BUILD_FOR_TUNING
          .def_readwrite("kernel_fine_control", &attn_options::kernel_fine_control)
#endif
        ;
#if AOTRITON_BUILD_FOR_TUNING
        // Expose KernelSlot enum
        py::enum_<attn_options::KernelSlot>(attn_options_class, "KernelSlot")
          .value("attn_fwd", attn_options::KernelSlot::attn_fwd)
          .value("debug_simulate_encoded_softmax", attn_options::KernelSlot::debug_simulate_encoded_softmax)
          .value("bwd_preprocess", attn_options::KernelSlot::bwd_preprocess)
          .value("bwd_preprocess_varlen", attn_options::KernelSlot::bwd_preprocess_varlen)
          .value("bwd_kernel_dk_dv", attn_options::KernelSlot::bwd_kernel_dk_dv)
          .value("bwd_kernel_dq", attn_options::KernelSlot::bwd_kernel_dq)
          .value("bwd_kernel_fuse", attn_options::KernelSlot::bwd_kernel_fuse)
          .value("MaxKernels", attn_options::KernelSlot::MaxKernels)
          .export_values()
        ;
#endif
        py::class_<attn_fwd_params>(m, "attn_fwd_params")
          .def(py::init<>())
#define RW(name) def_readwrite(#name, &attn_fwd_params::name)
          .RW(Q)
          .RW(K)
          .RW(V)
          .RW(B)
          .RW(Sm_scale)
          .RW(L)
          .RW(Out)
          .RW(cu_seqlens_q)
          .RW(cu_seqlens_k)
          .RW(Max_seqlen_q)
          .RW(Max_seqlen_k)
          .RW(seq_strides_q)
          .RW(seq_strides_k)
          .RW(dropout_p)
          .RW(philox_seed_ptr)
          .RW(philox_offset1)
          .RW(philox_offset2)
          .RW(philox_seed_output)
          .RW(philox_offset_output)
          .RW(encoded_softmax)
          .RW(persistent_atomic_counter)
          .RW(causal_type)
          .RW(varlen_type)
          .RW(window_left)
          .RW(window_right)
#undef RW
          .def_readonly_static("kVersion", &attn_fwd_params::kVersion)
        ;
        py::class_<attn_bwd_params>(m, "attn_bwd_params")
          .def(py::init<>())
#define RW(name) def_readwrite(#name, &attn_bwd_params::name)
          .RW(Q)
          .RW(K)
          .RW(V)
          .RW(B)
          .RW(Sm_scale)
          .RW(Out)
          .RW(DO)
          .RW(DK)
          .RW(DV)
          .RW(DQ)
          .RW(DB)
          .RW(L)
          .RW(D)
          .RW(cu_seqlens_q)
          .RW(cu_seqlens_k)
          .RW(Max_seqlen_q)
          .RW(Max_seqlen_k)
          .RW(seq_strides_q)
          .RW(seq_strides_k)
          .RW(dropout_p)
          .RW(philox_seed_ptr)
          .RW(philox_offset1)
          .RW(philox_offset2)
          .RW(causal_type)
          .RW(varlen_type)
          .RW(window_left)
          .RW(window_right)
          .RW(DQ_ACC)
#undef RW
          .def_readonly_static("kVersion", &attn_bwd_params::kVersion)
        ;
        m.def("attn_fwd",
              &aotriton::v3::flash::attn_fwd,
              "Flash Attention Operator Forward Pass",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("params"),
              py::arg("params_version"),
              py::arg("stream") = nullptr,
              py::arg("options") = nullptr);
        m.def("attn_bwd",
              &aotriton::v3::flash::attn_bwd,
              "Flash Attention Operator Backward Pass",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("params"),
              py::arg("params_version"),
              py::arg("stream") = nullptr,
              py::arg("options") = nullptr);
      }
  } // namespace pyaotriton::v3::flash

  // pyaotriton::v3 -> pyaotriton.v3
  void setup_module(py::module_& m) {
#if AOTRITON_BUILD_FOR_TUNING
    // Expose KernelControl struct with shared_ptr holder for reference semantics
    py::class_<aotriton::v3::KernelControl, std::shared_ptr<aotriton::v3::KernelControl>>(m, "KernelControl")
      .def(py::init<>())
      .def_readwrite("control_bits", &aotriton::v3::KernelControl::control_bits)
      .def_readwrite("hsaco_index", &aotriton::v3::KernelControl::hsaco_index)
      .def_readonly("total_hsacos", &aotriton::v3::KernelControl::total_hsacos)
      .def_readonly("kernel_psels", &aotriton::v3::KernelControl::kernel_psels)
      .def_readonly("kernel_copts", &aotriton::v3::KernelControl::kernel_copts)
      // Note: kernel_image is a raw pointer, typically exposed to Python as an integer
      // address. Its lifetime is tied to the underlying AOTRITON objects. Prefer using
      // get_kernel_image() to obtain a safe copy of the image as bytes.
      .def_readonly("kernel_image", &aotriton::v3::KernelControl::kernel_image)
      .def_readonly("image_size", &aotriton::v3::KernelControl::image_size)
      .def("get_kernel_image",
           [](const aotriton::v3::KernelControl &self) {
             if (self.kernel_image == nullptr || self.image_size == 0) {
               return py::bytes();
             }
             return py::bytes(
                 static_cast<const char *>(self.kernel_image),
                 static_cast<std::size_t>(self.image_size));
           },
           R"(Return a copy of the kernel image as a Python bytes object.

This function copies `image_size` bytes from the underlying kernel_image buffer
into a new Python-owned bytes object. The returned data remains valid even if
the original KernelControl instance or its backing resources are destroyed.)")
      .def_readonly_static("Default", &aotriton::v3::KernelControl::Default)
      .def_readonly_static("Ignore", &aotriton::v3::KernelControl::Ignore)
      .def_readonly_static("Manual", &aotriton::v3::KernelControl::Manual)
      .def_readonly_static("Skip", &aotriton::v3::KernelControl::Skip)
      .def_readonly_static("Query", &aotriton::v3::KernelControl::Query)
      .def_readonly_static("ExtractImage", &aotriton::v3::KernelControl::ExtractImage)
      ;

    // Expose KernelFineControl with array-like interface
    py::class_<aotriton::v3::KernelFineControl>(m, "KernelFineControl")
      .def("__getitem__", &aotriton::v3::KernelFineControl::at)
      .def("__len__", &aotriton::v3::KernelFineControl::size)
      ;
#endif
    py::module_ mod_flash = m.def_submodule("flash", "Flash Attention Operators");
    flash::setup_module(mod_flash);
  }

} // namespace pyaotriton::v3

namespace pyaotriton {
  namespace {
    // Self-register the v3 API submodule (see submodule_registry.h).
    const SubmoduleRegistrar _v3_registrar("v3", "v3 API namespace", &v3::setup_module);
  } // namespace
} // namespace pyaotriton
