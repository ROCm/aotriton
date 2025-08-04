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
        py::class_<attn_options>(m, "attn_options")
          .def(py::init<>())
          .def_readwrite("force_backend_index", &attn_options::force_backend_index)
        ;
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
        m.def("aiter_bwd",
              &aotriton::v3::flash::aiter_bwd,
              "Flash Attention Operator Backward Pass using AITER ASM Kernel.",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("params"),
              py::arg("params_version"),
              py::arg("stream") = nullptr,
              py::arg("options") = nullptr);
      }
  } // namespace pyaotriton::v3::flash
  void setup_module(py::module_& m) {
    // TODO: Optune
    py::module_ mod_flash = m.def_submodule("flash", "Flash Attention Operators");
    flash::setup_module(mod_flash);
  }
} // namespace pyaotriton::v3
