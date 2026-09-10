// Copyright © 2023-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../flyc.[[shim_kernel_name]].h"
#include <aotriton/_internal/triton_kernel.h>
#include <aotriton/_internal/kernel_cluster.h>
#include <aotriton/_internal/log.h>
#include <assert.h>
#include <string_view>

#define CURRENT_ENTRY_PUBLIC Flytune_[[shim_kernel_name]]__A[[arch_number]]__F[[godel_number]]

namespace { // Anonymous namespace

using namespace std::literals::string_view_literals;

// flyc has no LUT and no PerfFields struct: every functional resolves to
// exactly one hsaco (there is no autotune choice among multiple candidate
// kernel images the way Triton's shim has), so this table entry is a
// degenerate single-candidate dispatcher. There is no
// `kernel_psels`/`kernel_copts` JSON string array either -- the perf/copt
// knob set lives in the single hsaco's `#P` section and is read back via
// `Pon` from `context.perf()` after `lookup_optimal()`, not duplicated here.

// u8R generates char8_t which is poorly supported almost everywhere.
constexpr pstring_view FLATZIP_PATH
#if defined(_WIN32)
{ LR"xyzw([[flatzip_path]])xyzw" };
#else
{ R"xyzw([[flatzip_path]])xyzw" };
#endif
constexpr std::string_view FUNC_NAME { R"xyzw([[func_name]])xyzw" };
constexpr std::string_view ARCH_NAME { R"xyzw([[arch_name]])xyzw" };

// Checksum can be confirmed with `echo -n '<string>' | b2sum -l 64`
static AOTRITON_NS::TritonKernelCompactMeta meta_list[] = {
    [[meta_hsacos]]
};

static constexpr int kTotalNumKernels = 1;

using AOTRITON_NS::v3::[[kernel_family_name]]::flytune::[[shim_kernel_name]]_packed_string;

static AOTRITON_NS::TritonKernelCluster<kTotalNumKernels>
kernel_cluster(meta_list,
               [[shim_kernel_name]]_packed_string);

}; // End of anonymous namespace

namespace AOTRITON_NS::v3::[[kernel_family_name]]::flytune {

// Returns the selected kernel_index. flyc builds exactly one hsaco per
// surviving functional, so this always selects index 0 -- there is no LUT
// lookup and no preferred-kernel override to consult. Disabled functionals
// never get an entry compiled at all: their .cc file is simply not emitted
// (FlycShimGenerator::create_sub_generator).
int CURRENT_ENTRY_PUBLIC([[context_class_name]]& context, int mod_number) {
    (void) mod_number;
    constexpr int kernel_index = 0;
    context.kernel_on_device = kernel_cluster.get(kernel_index);
    context.pp_args_index = [[deduplicated_pp_args_function_index]];
    context.flatzip_path = FLATZIP_PATH;
    // aks2_entry IS the func_name (supplied by f.unified_signature)
    context.aks2_entry   = FUNC_NAME;
    context.func_name = FUNC_NAME;
    context.arch_name = ARCH_NAME;
#if AOTRITON_BUILD_FOR_TUNING
    context._total_number_of_kernels = kTotalNumKernels;
#endif
    return kernel_index;
}

#undef CURRENT_ENTRY_PUBLIC
}

// [[human_readable_signature]]

// [[sql]]

// vim: set fileencoding=utf-8
