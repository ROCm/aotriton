// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <aotriton/config.h>
#include <aotriton/dtypes.h>
#include <aotriton/flash.h>
#include <aotriton/flash/backends.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/cpp_tune.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include "submodule_registry.h"
// varlen_to_wire is internal: a caller fills a VarlenBits and never sees the
// word. Bound anyway so the round-trip gate can assert the encoding from
// Python, which is where the bit-field bindings themselves are exercised.
#include "../csrc/varlen.h"

namespace nb = nanobind;
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
      void setup_module(nb::module_& m) {
        // nanobind CANNOT bind a bit-field through def_rw: that needs
        // &Struct::member, and a pointer-to-member to a bit-field is ill-formed
        // -- you cannot take the address of one in C++ at all. So every BIT-FIELD
        // below goes through def_prop_rw with lambdas that read and write
        // THROUGH the bit-field rather than around it. A BITF macro beside the
        // RW macro below, for the same reason RW exists: the field list should
        // read as a list.
        //
        // One class per side-mode, then VarlenBits holding two of them. The two
        // modes are ordinary members of class type, not bit-fields, so they DO
        // take def_rw -- which on a registered class type returns by
        // reference (reference_internal), so `params.varlen_bits.qmode.stacked
        // = 1` reaches the parent; a copy would make every write vanish
        // silently.
        nb::class_<VarlenMode>(m, "VarlenMode")
          .def(nb::init<>())
#define BITF(name) def_prop_rw(#name,                                       \
            [](const VarlenMode& v) { return uint32_t(v.name); },            \
            [](VarlenMode& v, uint32_t x) { v.name = x; })
          .BITF(stacked)
          .BITF(length)
          .BITF(position)
#undef BITF
        ;
        nb::class_<VarlenBits>(m, "VarlenBits")
          .def(nb::init<>())
          .def_rw("qmode", &VarlenBits::qmode)
          .def_rw("kmode", &VarlenBits::kmode)
          // lse_layout IS a bit-field, so it takes the def_prop_rw route above.
          .def_prop_rw("lse_layout",
                        [](const VarlenBits& v) { return uint32_t(v.lse_layout); },
                        [](VarlenBits& v, uint32_t x) { v.lse_layout = x; })
          .def_prop_rw("principal_diagonal_choice",
                       [](const VarlenBits& v) { return uint32_t(v.principal_diagonal_choice); },
                       [](VarlenBits& v, uint32_t x) { v.principal_diagonal_choice = x; })
        ;
        // The axis-value constants, so Python can spell a configuration rather
        // than assemble one. nb::enum_ (KernelSlot, below) is the local
        // precedent but does not apply: these are structs of static constexpr,
        // deliberately not enums (see the note in flash.h), so each becomes a
        // namespace-like class carrying read-only statics.
        {
          using namespace aotriton::v3::flash;
          nb::class_<VarlenStacked>(m, "VarlenStacked")
            .def_ro_static("BHSD", &VarlenStacked::BHSD)
            .def_ro_static("THD", &VarlenStacked::THD);
          nb::class_<VarlenLength>(m, "VarlenLength")
            .def_ro_static("MAX", &VarlenLength::MAX)
            .def_ro_static("CUMULATIVE", &VarlenLength::CUMULATIVE)
            .def_ro_static("INDIVIDUAL", &VarlenLength::INDIVIDUAL);
          nb::class_<VarlenPosition>(m, "VarlenPosition")
            .def_ro_static("IMPLIED", &VarlenPosition::IMPLIED)
            .def_ro_static("REUSE", &VarlenPosition::REUSE)
            .def_ro_static("ARRAY", &VarlenPosition::ARRAY);
          nb::class_<VarlenLseLayout>(m, "VarlenLseLayout")
            .def_ro_static("HT", &VarlenLseLayout::HT)
            .def_ro_static("TH", &VarlenLseLayout::TH);
          nb::class_<VarlenPrincipalDiagonal>(m, "VarlenPrincipalDiagonal")
            .def_ro_static("TOP_LEFT", &VarlenPrincipalDiagonal::TOP_LEFT)
            .def_ro_static("BOTTOM_RIGHT", &VarlenPrincipalDiagonal::BOTTOM_RIGHT);
          // So a test can assert the wire value from Python as well as from the
          // static_assert table in csrc/varlen.h. This is the only place the
          // binding layer's correctness is observable: a bit-field bound with the
          // wrong WIDTH still reads back whatever was written, as long as it
          // fits.
          m.def("varlen_to_wire",
                [](const VarlenBits& v) {
                  return aotriton::v3::flash::internal::varlen_to_wire(v);
                },
                nb::arg("varlen_bits"));
        }
        // Backend-index constants, from the generated
        // <aotriton/flash/backends.h>. Plain int attributes rather than
        // nb::enum_, for the reason modules/flash/tests/aotriton_flash.py
        // records about CausalType: reading an enum's integer needs `.value`,
        // which makes every call site verbose. These feed
        // attn_options.force_backend_index, which is an int.
        //
        // The constant NAMES are not written here -- they expand from the
        // generator's X-macro, so this binding cannot fall out of step with
        // BackendEnum the way the hand-copied CausalType did. Only the two
        // STRUCT names are named, and adding an operator is what adds one.
#define AOTRITON_BIND_BACKENDS(Struct)                                        \
        {                                                                     \
          using T = aotriton::v3::flash::Struct;                              \
          nb::class_<T> c(m, #Struct);                                        \
          nb::dict by_index;                                                  \
          AOTRITON_BACKENDS_##Struct(AOTRITON_BIND_ONE)                       \
          c.attr("Max") = T::Max;                                             \
          /* index -> the name @ati.backend declared, so a caller can pin a   \
             backend by the description's vocabulary instead of an integer. */\
          c.attr("by_index") = by_index;                                      \
        }
#define AOTRITON_BIND_ONE(Name, Str)                                          \
        c.attr(#Name) = T::Name;                                              \
        by_index[nb::int_(T::Name)] = Str;
        AOTRITON_BIND_BACKENDS(OpAttnFwdBackend)
        AOTRITON_BIND_BACKENDS(OpAttnBwdBackend)
#undef AOTRITON_BIND_ONE
#undef AOTRITON_BIND_BACKENDS

        auto attn_options_class = nb::class_<attn_options>(m, "attn_options")
          .def(nb::init<>())
          .def_rw("force_backend_index", &attn_options::force_backend_index)
          .def_rw("deterministic", &attn_options::deterministic)
#if AOTRITON_BUILD_FOR_TUNING
          .def_rw("kernel_fine_control", &attn_options::kernel_fine_control)
#endif
        ;
#if AOTRITON_BUILD_FOR_TUNING
        // Expose KernelSlot enum
        nb::enum_<attn_options::KernelSlot>(attn_options_class, "KernelSlot", nb::is_arithmetic())
          .value("attn_fwd", attn_options::KernelSlot::attn_fwd)
          .value("debug_simulate_encoded_softmax", attn_options::KernelSlot::debug_simulate_encoded_softmax)
          .value("bwd_preprocess", attn_options::KernelSlot::bwd_preprocess)
          .value("bwd_kernel_dk_dv", attn_options::KernelSlot::bwd_kernel_dk_dv)
          .value("bwd_kernel_dq", attn_options::KernelSlot::bwd_kernel_dq)
          .value("bwd_kernel_fuse", attn_options::KernelSlot::bwd_kernel_fuse)
          .value("flyc_attn_fwd", attn_options::KernelSlot::flyc_attn_fwd)
          .value("flyc_bwd_dkdv", attn_options::KernelSlot::flyc_bwd_dkdv)
          .value("flyc_bwd_dq", attn_options::KernelSlot::flyc_bwd_dq)
          .value("MaxKernels", attn_options::KernelSlot::MaxKernels)
          .export_values()
        ;
#endif
        nb::class_<attn_fwd_params>(m, "attn_fwd_params")
          .def(nb::init<>())
#define RW(name) def_rw(#name, &attn_fwd_params::name)
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
          // def_rw on a registered class type returns the member by
          // reference (return_value_policy::reference_internal), so
          // `params.varlen_bits.qmode.stacked = 1` mutates the parent in place --
          // had it returned a copy, every write from Python would silently
          // vanish and a varlen request would run as dense.
          .RW(varlen_bits)
#undef RW
          .def_ro_static("kVersion", &attn_fwd_params::kVersion)
        ;
        nb::class_<attn_bwd_params>(m, "attn_bwd_params")
          .def(nb::init<>())
#define RW(name) def_rw(#name, &attn_bwd_params::name)
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
          .def_ro_static("kVersion", &attn_bwd_params::kVersion)
        ;
        m.def("attn_fwd",
              &aotriton::v3::flash::attn_fwd,
              "Flash Attention Operator Forward Pass",
              nb::call_guard<nb::gil_scoped_release>(),
              nb::arg("params"),
              nb::arg("params_version"),
              nb::arg("stream") = nullptr,
              nb::arg("options").none() = nullptr);
        m.def("attn_bwd",
              &aotriton::v3::flash::attn_bwd,
              "Flash Attention Operator Backward Pass",
              nb::call_guard<nb::gil_scoped_release>(),
              nb::arg("params"),
              nb::arg("params_version"),
              nb::arg("stream") = nullptr,
              nb::arg("options").none() = nullptr);
      }
  } // namespace pyaotriton::v3::flash

  // pyaotriton::v3 -> pyaotriton.v3
  void setup_module(nb::module_& m) {
#if AOTRITON_BUILD_FOR_TUNING
    // Expose KernelControl struct. KernelFineControl hands it out as a
    // std::shared_ptr, which nanobind's shared_ptr caster (stl/shared_ptr.h)
    // wraps without copying, so writes from Python reach the original.
    nb::class_<aotriton::v3::KernelControl>(m, "KernelControl")
      .def(nb::init<>())
      .def_rw("control_bits", &aotriton::v3::KernelControl::control_bits)
      .def_rw("hsaco_index", &aotriton::v3::KernelControl::hsaco_index)
      .def_ro("total_hsacos", &aotriton::v3::KernelControl::total_hsacos)
      .def_ro("kernel_psels", &aotriton::v3::KernelControl::kernel_psels)
      .def_ro("kernel_copts", &aotriton::v3::KernelControl::kernel_copts)
      // Note: kernel_image is a raw pointer, typically exposed to Python as an integer
      // address. Its lifetime is tied to the underlying AOTRITON objects. Prefer using
      // get_kernel_image() to obtain a safe copy of the image as bytes.
      .def_prop_ro("kernel_image",
                   [](const aotriton::v3::KernelControl &self) {
                     // nanobind casts void*, not const void*
                     return const_cast<void *>(self.kernel_image);
                   })
      .def_ro("image_size", &aotriton::v3::KernelControl::image_size)
      .def("get_kernel_image",
           [](const aotriton::v3::KernelControl &self) {
             if (self.kernel_image == nullptr || self.image_size == 0) {
               return nb::bytes("", 0);  // nb::bytes() would be a null object
             }
             return nb::bytes(
                 static_cast<const char *>(self.kernel_image),
                 static_cast<std::size_t>(self.image_size));
           },
           R"(Return a copy of the kernel image as a Python bytes object.

This function copies `image_size` bytes from the underlying kernel_image buffer
into a new Python-owned bytes object. The returned data remains valid even if
the original KernelControl instance or its backing resources are destroyed.)")
      .def_ro_static("Default", &aotriton::v3::KernelControl::Default)
      .def_ro_static("Ignore", &aotriton::v3::KernelControl::Ignore)
      .def_ro_static("Manual", &aotriton::v3::KernelControl::Manual)
      .def_ro_static("Skip", &aotriton::v3::KernelControl::Skip)
      .def_ro_static("Query", &aotriton::v3::KernelControl::Query)
      .def_ro_static("ExtractImage", &aotriton::v3::KernelControl::ExtractImage)
      ;

    // Expose KernelFineControl with array-like interface
    nb::class_<aotriton::v3::KernelFineControl>(m, "KernelFineControl")
      .def("__getitem__", &aotriton::v3::KernelFineControl::at)
      .def("__len__", &aotriton::v3::KernelFineControl::size)
      ;
#endif
    nb::module_ mod_flash = m.def_submodule("flash", "Flash Attention Operators");
    flash::setup_module(mod_flash);
  }

} // namespace pyaotriton::v3

namespace pyaotriton {
  namespace {
    // Self-register the v3 API submodule (see submodule_registry.h).
    const SubmoduleRegistrar _v3_registrar("v3", "v3 API namespace", &v3::setup_module);
  } // namespace
} // namespace pyaotriton
