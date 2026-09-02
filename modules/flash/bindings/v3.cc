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
// varlen_to_wire is internal: a caller fills a VarlenBits and never sees the
// word. Bound anyway so the round-trip gate can assert the encoding from
// Python, which is where the bit-field bindings themselves are exercised.
#include "../csrc/varlen.h"

namespace py = pybind11;
#if AOTRITON_ENABLE_SUFFIX
namespace aotriton = AOTRITON_NS;
#endif

namespace pyaotriton::v3 {
  namespace flash {
      using aotriton::v3::flash::attn_fwd_params;
      using aotriton::v3::flash::attn_bwd_params;
      using aotriton::v3::flash::attn_options;
      using aotriton::v3::flash::VarlenBits;
      using aotriton::v3::flash::VarlenMode;
      void setup_module(py::module_& m) {
        // pybind11 CANNOT bind a bit-field through def_readwrite: that needs
        // &Struct::member, and a pointer-to-member to a bit-field is ill-formed
        // -- you cannot take the address of one in C++ at all. So every field of
        // VarlenBits goes through def_property with lambdas that read and write
        // THROUGH the bit-field rather than around it. A BITF macro beside the
        // RW macro below, for the same reason RW exists: the field list should
        // read as a list.
        // One class per side-mode, then VarlenBits holding two of them.
        // def_readwrite on a registered class type returns by reference
        // (reference_internal), so `params.varlen_bits.qmode.stacked = 1`
        // reaches the parent; a copy would make every write vanish silently.
        py::class_<VarlenMode>(m, "VarlenMode")
          .def(py::init<>())
#define BITF(name) def_property(#name,                                       \
            [](const VarlenMode& v) { return uint32_t(v.name); },            \
            [](VarlenMode& v, uint32_t x) { v.name = x; })
          .BITF(stacked)
          .BITF(length)
          .BITF(position)
#undef BITF
        ;
        py::class_<VarlenBits>(m, "VarlenBits")
          .def(py::init<>())
          .def_readwrite("qmode", &VarlenBits::qmode)
          .def_readwrite("kmode", &VarlenBits::kmode)
          // pybind11 CANNOT bind a bit-field through def_readwrite: that needs
          // &Struct::member, and a pointer-to-member to a bit-field is
          // ill-formed. lse_layout therefore goes through def_property.
          .def_property("lse_layout",
                        [](const VarlenBits& v) { return uint32_t(v.lse_layout); },
                        [](VarlenBits& v, uint32_t x) { v.lse_layout = x; })
        ;
        // The axis-value constants, so Python can spell a configuration rather
        // than assemble one. py::enum_ (KernelSlot, below) is the local
        // precedent but does not apply: these are structs of static constexpr,
        // deliberately not enums (see the note in flash.h), so each becomes a
        // namespace-like class carrying read-only statics.
        {
          using namespace aotriton::v3::flash;
          py::class_<VarlenStacked>(m, "VarlenStacked")
            .def_readonly_static("BHSD", &VarlenStacked::BHSD)
            .def_readonly_static("THD", &VarlenStacked::THD);
          py::class_<VarlenLength>(m, "VarlenLength")
            .def_readonly_static("MAX", &VarlenLength::MAX)
            .def_readonly_static("CUMULATIVE", &VarlenLength::CUMULATIVE)
            .def_readonly_static("INDIVIDUAL", &VarlenLength::INDIVIDUAL);
          py::class_<VarlenPosition>(m, "VarlenPosition")
            .def_readonly_static("IMPLIED", &VarlenPosition::IMPLIED)
            .def_readonly_static("REUSE", &VarlenPosition::REUSE)
            .def_readonly_static("ARRAY", &VarlenPosition::ARRAY);
          py::class_<VarlenLseLayout>(m, "VarlenLseLayout")
            .def_readonly_static("HT", &VarlenLseLayout::HT)
            .def_readonly_static("TH", &VarlenLseLayout::TH);
          // So a test can assert the wire value from Python as well as from the
          // static_assert table in csrc/varlen.h. This is the only place the
          // pybind layer's correctness is observable: a bit-field bound with the
          // wrong WIDTH still reads back whatever was written, as long as it
          // fits.
          m.def("varlen_to_wire",
                [](const VarlenBits& v) {
                  return aotriton::v3::flash::internal::varlen_to_wire(v);
                },
                py::arg("varlen_bits"));
        }
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
          .RW(seqinfo_q0)
          .RW(seqinfo_k0)
          .RW(Max_seqlen_q)
          .RW(Max_seqlen_k)
          .RW(seqinfo_q1)
          .RW(seqinfo_k1)
          .RW(dropout_p)
          .RW(philox_seed_ptr)
          .RW(philox_offset1)
          .RW(philox_offset2)
          .RW(philox_seed_output)
          .RW(philox_offset_output)
          .RW(encoded_softmax)
          .RW(persistent_atomic_counter)
          .RW(causal_type)
          .RW(window_left)
          .RW(window_right)
          // An ordinary member of class type, not a bit-field, so RW compiles.
          // def_readwrite on a registered class type returns the member by
          // reference (return_value_policy::reference_internal), so
          // `params.varlen_bits.q_stacked = 1` mutates the parent in place --
          // had it returned a copy, every write from Python would silently
          // vanish and a varlen request would run as dense.
          .RW(varlen_bits)
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
          .RW(seqinfo_q0)
          .RW(seqinfo_k0)
          .RW(Max_seqlen_q)
          .RW(Max_seqlen_k)
          .RW(seqinfo_q1)
          .RW(seqinfo_k1)
          .RW(dropout_p)
          .RW(philox_seed_ptr)
          .RW(philox_offset1)
          .RW(philox_offset2)
          .RW(causal_type)
          .RW(window_left)
          .RW(window_right)
          .RW(DQ_ACC)
          .RW(varlen_bits)   // see attn_fwd_params above
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
