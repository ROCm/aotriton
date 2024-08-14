// Copyright © 2023-2024 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/cpp_tune.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace pyaotriton {
  namespace v2 {
    namespace flash {
      using aotriton::v2::flash::FwdExtraArguments;
      using aotriton::v2::flash::BwdExtraArguments;
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
        m.def("attn_fwd",
              &aotriton::v2::flash::attn_fwd,
              "Flash Attention Forward Pass",
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
              py::arg("encoded_softmax"),
              py::arg("is_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = FwdExtraArguments());
        m.def("attn_fwd_compact_varlen",
              &aotriton::v2::flash::attn_fwd_compact_varlen,
              "Flash Attention Forward Pass, Compact Stored Varlen",
              py::arg("q"),
              py::arg("k"),
              py::arg("v"),
              py::arg("cu_seqlens_q"),
              py::arg("cu_seqlens_k"),
              py::arg("max_seqlen_q"),
              py::arg("max_seqlen_k"),
              py::arg("b"),
              py::arg("sm_scale"),
              py::arg("softmax_lse"),
              py::arg("out"),
              py::arg("dropout_p"),
              py::arg("philox_seed"),
              py::arg("philox_offset1"),
              py::arg("philox_offset2"),
              py::arg("encoded_softmax"),
              py::arg("is_causal"),
              py::arg("stream") = nullptr,
              py::arg("extargs") = FwdExtraArguments());
        m.def("attn_bwd",
              &aotriton::v2::flash::attn_bwd,
              "Flash Attention Backward Pass",
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
        m.def("attn_bwd_compact_varlen",
              &aotriton::v2::flash::attn_bwd_compact_varlen,
              "Flash Attention Backward Pass, Compact Stored Varlen",
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
        m.def("debug_fill_dropout_rng",
              &aotriton::v2::flash::debug_fill_dropout_rng,
              "Flash Attention Debugging Function to get raw RNG numbers used in dropout",
              py::arg("q"),
              py::arg("philox_seed"),
              py::arg("philox_offset"),
              py::arg("stream") = nullptr);
        m.def("debug_fill_dropout_rng_tensor",
              &aotriton::v2::flash::debug_fill_dropout_rng_tensor,
              "Flash Attention Debugging Function to get raw RNG numbers used in dropout",
              py::arg("q"),
              py::arg("philox_seed"),
              py::arg("philox_offset"),
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
    // FIXME: deduplication of T0 code
    py::class_<aotriton::TensorView<0>>(m, "T0")
      .def(py::init<intptr_t, aotriton::DType>())
      .def("size", &aotriton::TensorView<0>::size)
      .def("stride", &aotriton::TensorView<0>::stride)
      .def_property_readonly("sizes", &aotriton::TensorView<0>::sizes)
      .def_property_readonly("strides", &aotriton::TensorView<0>::strides)
      .def_property_readonly("data_ptr", &aotriton::TensorView<0>::data_ptr)
      .def_property_readonly("dtype", &aotriton::TensorView<0>::dtype);
    py::module_ mod_v2api = m.def_submodule("v2", "v2 API namespace");
    v2::setup_module(mod_v2api);
  }

} // namespace pyaotriton

PYBIND11_MODULE(pyaotriton, m) {
  pyaotriton::setup_module(m);
}
