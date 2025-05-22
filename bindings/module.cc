// Copyright © 2023-2025 Advanced Micro Devices, Inc.
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
#include <string>

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton {
  namespace v2 {
    namespace flash {
      using aotriton::v2::flash::FwdExtraArguments;
      using aotriton::v2::flash::BwdExtraArguments;
      using aotriton::v2::flash::FusedBwdExtraArguments;
      void setup_module(py::module_& m) {
        m.def("check_gpu", &aotriton::v2::flash::check_gpu, py::arg("stream"));
        py::class_<FwdExtraArguments, aotriton::v2::CppTune>(m, "FwdExtraArguments")
          .def(py::init<>())
        ;
        py::class_<BwdExtraArguments>(m, "BwdExtraArguments")
          .def(py::init<>())
#if AOTRITON_BUILD_FOR_TUNING
          .def_readwrite("dkdv", &BwdExtraArguments::dkdv)
          .def_readwrite("dqdb", &BwdExtraArguments::dqdb)
#endif
        ;
        py::class_<FusedBwdExtraArguments, aotriton::v2::CppTune>(m, "FusedBwdExtraArguments")
          .def(py::init<>())
        ;
        m.def("attn_fwd",
              &aotriton::v2::flash::attn_fwd,
              "Flash Attention Forward Pass",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("b"),
              py::arg("sm_scale"),
              py::arg("softmax_lse"),
              py::arg("out"),
              py::arg("dropout_p"),
              py::arg("philox_seed"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("philox_seed_output"),
              py::arg("philox_offset_output"),
              py::arg("encoded_softmax"),
              py::arg("is_causal"),
              py::arg("atomic_for_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = FwdExtraArguments());
        m.def("attn_fwd_compact_varlen",
              &aotriton::v2::flash::attn_fwd_compact_varlen,
              "Flash Attention Forward Pass, Compact Stored Varlen",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("b"),
              py::arg("cu_seqlens_q"),
              py::arg("cu_seqlens_k"),
              py::arg("max_seqlen_q"),
              py::arg("max_seqlen_k"),
              py::arg("sm_scale"),
              py::arg("softmax_lse"),
              py::arg("out"),
              py::arg("dropout_p"),
              py::arg("philox_seed"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("philox_seed_output"),
              py::arg("philox_offset_output"),
              py::arg("encoded_softmax"),
              py::arg("is_causal"),
              py::arg("atomic_for_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = FwdExtraArguments());
        m.def("attn_bwd",
              &aotriton::v2::flash::attn_bwd,
              "Flash Attention Backward Pass",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("b"),
              py::arg("sm_scale"),
              py::arg("out"),
              py::arg("dout"),
              py::arg("dq"),
              py::arg("dk"),
              py::arg("dv"),
              py::arg("db"),
              py::arg("softmax_lse"),
              py::arg("delta"),
              py::arg("dropout_p"),
              py::arg("philox_seed"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("is_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = BwdExtraArguments());
        m.def("attn_bwd_fused",
              &aotriton::v2::flash::attn_bwd_fused,
              "Flash Attention Backward Pass",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("b"),
              py::arg("sm_scale"),
              py::arg("out"),
              py::arg("dout"),
              py::arg("dq"),
              py::arg("dk"),
              py::arg("dv"),
              py::arg("db"),
              py::arg("softmax_lse"),
              py::arg("dropout_p"),
              py::arg("philox_seed"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("is_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = FusedBwdExtraArguments());
        m.def("attn_bwd_compact_varlen",
              &aotriton::v2::flash::attn_bwd_compact_varlen,
              "Flash Attention Backward Pass, Compact Stored Varlen",
              py::call_guard<py::gil_scoped_release>(),
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("cu_seqlens_q"),
              py::arg("cu_seqlens_k"),
              py::arg("max_seqlen_q"),
              py::arg("max_seqlen_k"),
              py::arg("b"),
              py::arg("sm_scale"),
              py::arg("out"),
              py::arg("dout"),
              py::arg("dq"),
              py::arg("dk"),
              py::arg("dv"),
              py::arg("db"),
              py::arg("softmax_lse"),
              py::arg("delta"),
              py::arg("dropout_p"),
              py::arg("philox_seed"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("is_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = BwdExtraArguments());
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
      using aotriton::v2::CppTune;
      py::class_<aotriton::v2::CppTune>(m, "CppTune")
          .def(py::init<>())
#if AOTRITON_BUILD_FOR_TUNING
          .def_readwrite("force_kernel_index", &CppTune::force_kernel_index)
          .def_readonly("total_number_of_kernels", &CppTune::total_number_of_kernels)
          .def_readonly("selected_kernel_psels", &CppTune::selected_kernel_psels)
          .def_readonly("selected_kernel_copts", &CppTune::selected_kernel_copts)
          .def_readwrite("peek_kernel_image", &CppTune::peek_kernel_image)
          .def("get_kernel_image",
               [](const CppTune* tune) {
                  std::string s((const char*)tune->kernel_image, tune->image_size);
                  return py::bytes(s);
               })
#endif
      ;
      using aotriton::v2::CppTuneSpecialKernelIndex;
#define EV(name) value(#name, aotriton::v2::CppTuneSpecialKernelIndex::name)
      py::enum_<CppTuneSpecialKernelIndex>(m, "CppTuneSpecialKernelIndex")
        .EV(kDefault)
        .EV(kSkipGPUCall)
        .export_values();
#undef EV
      py::module_ mod_flash = m.def_submodule("flash", "Flash Attention API");
      flash::setup_module(mod_flash);
    }
  } // namespace v2

  namespace v3 {
    void setup_module(py::module_& m); // Impl. goes into v3.cc
  } // namespace v3

  void def_stream(py::module_& m) {
    py::class_<aotriton::Stream>(m, "Stream").def(py::init<>());
  }

  void def_dtype(py::module_& m) {
#define EV(name) value(#name, aotriton::DType::name)
    py::enum_<aotriton::DType>(m, "DType")
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

  void def_hipruntime(py::module_& m);
  void def_hipmemory(py::module_& m);

  template<int Rank>
  void def_tensorview(py::module_& m, const std::string& name) {
    py::class_<aotriton::TensorView<Rank>>(m, name.c_str())
      .def(py::init<intptr_t, std::array<uint64_t, Rank>, std::array<uint64_t, Rank>, aotriton::DType>())
      .def("size", &aotriton::TensorView<Rank>::size)
      .def("stride", &aotriton::TensorView<Rank>::stride)
      .def_property_readonly("sizes", &aotriton::TensorView<Rank>::sizes)
      .def_property_readonly("strides", &aotriton::TensorView<Rank>::strides)
      .def_property_readonly("data_ptr", &aotriton::TensorView<Rank>::data_ptr)
      .def_property_readonly("dtype", &aotriton::TensorView<Rank>::dtype);
  }

  void setup_module(py::module_& m) {
    m.doc() = "AOTriton Python binding";
    def_stream(m);
    def_dtype(m);
    def_hipruntime(m);
    def_tensorview<4>(m, "T4");
    def_tensorview<2>(m, "T2");
    def_tensorview<1>(m, "T1");
    def_hipmemory(m);
    // FIXME: deduplication of T0 code
    py::class_<aotriton::TensorView<0>>(m, "T0")
      .def(py::init<intptr_t, aotriton::DType>())
      .def("size", &aotriton::TensorView<0>::size)
      .def("stride", &aotriton::TensorView<0>::stride)
      .def_property_readonly("sizes", &aotriton::TensorView<0>::sizes)
      .def_property_readonly("strides", &aotriton::TensorView<0>::strides)
      .def_property_readonly("data_ptr", &aotriton::TensorView<0>::data_ptr)
      .def_property_readonly("dtype", &aotriton::TensorView<0>::dtype);
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
    py::module_ mod_v2api = m.def_submodule("v2", "v2 API namespace");
    v2::setup_module(mod_v2api);
    py::module_ mod_v3api = m.def_submodule("v3", "v3 API namespace");
    v3::setup_module(mod_v3api);
  }

} // namespace pyaotriton

PYBIND11_MODULE(pyaotriton, m) {
  pyaotriton::setup_module(m);
}
