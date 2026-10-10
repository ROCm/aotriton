// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#pragma once

#include <aotriton/config.h>
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/pon.h>
#include <aotriton/_internal/log.h>
#include <aotriton/dtypes.h>
#include <aotriton/runtime.h>
#include <aotriton/util.h>
#include <aotriton/_internal/lazy_tensor_internal.h>
#include <functional>
#include <string>
#include <vector>

#if [[shared_iface]]
namespace AOTRITON_NS::v3::[[shared_iface_family]] {
    struct [[param_class_name]];
    struct [[call_options_struct]];
}
#endif

namespace AOTRITON_NS::v3::[[kernel_family_name]] {

#if [[shared_iface]]
using AOTRITON_NS::v3::[[shared_iface_family]]::[[param_class_name]];
#else
// The parameter class must be defined here when
// There is no common operator for [[shim_kernel_name]].
struct [[param_class_name]] {
    [[func_fields]];
};
#endif

struct [[context_class_name]] {
    const [[param_class_name]] *params = nullptr;
#if [[shared_iface]]
    const [[call_options_struct]] *call_options = nullptr;
#endif
    template <typename ParentContext>
    [[context_class_name]](const ParentContext& pcontext, bool condition)
      : launch_condition(condition)
    {
        params = pcontext.params;
#if [[shared_iface]]
        call_options = pcontext.call_options;
#endif
    }
    // flyc has no perf struct: the perf/copt knob set is a PON (Plain / Python
    // Object Notation) ';'-separated string, not a C struct. `perf_` is
    // populated once in lookup_optimal(), after the image is chosen and before
    // any launch, so grid_calculator() never parses on the launch path.
    Pon perf_;
    // Context helpers run EARLIER in lookup_optimal(), on the documented (and
    // expiring) assumption that none of them needs perf() -- see the
    // context-helper-evaluate block in lookup_optimal() (flyc.cc). Breaking
    // that assumption is otherwise silent: the helper would read a
    // default-constructed Pon and get_int/get_bool would quietly take their
    // fallback. Make the first violation loud instead, with a log line rather
    // than just a wrong answer.
    bool perf_populated_ = false;
    Pon perf() const {
        if (!perf_populated_) {
            AOTRITON_LOG(LOG_WARNING,
                         "[[shim_kernel_name]] perf() read before perf_ is populated -- "
                         "a context helper must not depend on perf() (see lookup_optimal())");
        }
        return perf_;
    }

    // The Gpu lookup_optimal(Gpu gpu) was called with. Context
    // helpers take no arguments by design -- see context_helper_declares
    // below -- so this is how one that needs the arch reaches it: call
    // get_archmod_number(current_gpu) itself. Not an arch_number/mod_number
    // pair, so a helper is free to ask either question rather than being
    // boxed into godel_number()'s. Set once, at the top of lookup_optimal(),
    // before any helper can run.
    Gpu current_gpu = GPU_ARCH_UNKNOWN;

    // Context helpers: host-side computations a plain operand rename cannot
    // express (`ati.context_helper(...)` on an @ati.scalar/@ati.tensor). The
    // generator only DECLARES these; modules/<family>/csrc/<kernel>.cc HAND
    // IMPLEMENTS them, the same split grid_calculator() already uses. Each
    // helper is called at most once per launch and cached into its scratch
    // member so `pp_args` can take its address: the vector holds pointers, and
    // a helper's return value has no other stable home for the duration of the
    // launch.
    [[context_helper_declares]]

    TritonKernel* kernel_on_device = nullptr;
    int pp_args_index = -1;
    pstring_view flatzip_path;
    std::string_view aks2_entry;
    std::string_view func_name;
    std::string_view arch_name;
    // Note to save ELF space, this object is constructed on the fly.
    const char* _debug_kernel_name = nullptr;
#if AOTRITON_BUILD_FOR_TUNING
    int _has_preferred_kernel = -1; // For C++ based autotune database generation
    int _total_number_of_kernels = -1;
    const char* _preferred_kernel_psels = nullptr;
    const char* _preferred_kernel_copts = nullptr;
    bool peek_kernel_image = false;
#endif
    bool launch_condition = true;

    hipError_t lookup_optimal(Gpu gpu);
    hipError_t launch(hipStream_t stream) const;

    dim3 grid_calculator() const;
    std::function<dim3(const [[context_class_name]]&)> custom_grid_calculator;

    int64_t godel_number() const;
    static std::tuple<int, int> get_archmod_number(Gpu gpu);
    static constexpr int kMaxGodelNumber = [[number_of_functionals]];

    // Returns the selected kernel_index (>= 0), or -1 when no valid kernel was
    // selected (in which case aks2_entry/func_name/arch_name are left unset).
    typedef int (*AutoTuneTableEntry)([[context_class_name]]& context, int mod_number);
    static AutoTuneTableEntry autotune_table[][ kMaxGodelNumber ];

    // The compiled rung table(s), arch-indexed exactly like autotune_table
    // above (same arch_number ordering). GENERATED from the same
    // surviving-functional set that fills autotune_table -- not
    // hand-maintained -- so it cannot silently drift from what was actually
    // compiled. Two parallel arrays,
    // not a struct, so no new type name has to be family-unique: one entry
    // per helper-wired, non-bool functional axis (today: BLOCK_DMODEL only;
    // PADDED_HEAD is derived from the rounding decision in the hand-written
    // helper below, not itself table-driven). Empty when this kernel wires
    // no such axis to a context helper.
    [[compiled_rung_table_declares]]

    // Mutable scratch storage for context-helper results: assigned once in
    // lookup_optimal(), then read by pointer from the kernarg vector that
    // pp_args builds. `mutable` because pp_args is a free function taking
    // `const [[context_class_name]]&` and cannot otherwise be given a context
    // whose scratch members are already written. Lifetime is the context's,
    // which outlives launch().
    [[context_helper_scratch_members]]
};

struct [[metadata_class_name]] {
    // Deliberately empty: this used to declare get_<axis>_choices() feature
    // tables (kernel.py's analogue still does, and its Triton binning site
    // still calls them) but nothing ever called the flyc side, and the table
    // it would report is the axis's declared choices, not this arch's
    // compiled subset -- see the context class's own compiled_<axis> /
    // compiled_<axis>_count arrays for the table that is actually correct and
    // actually used.
};

namespace flytune {

extern const char [[shim_kernel_name]]_packed_string[];

[[declare_list_of_deduplicated_lut_functions]]

[[kernel_table_entry_declares]]

}


}

// vim: set fileencoding=utf-8
