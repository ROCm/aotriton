// Copyright © 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// clang-format off
#include "iface.[[iface_name]].h"
#include <aotriton/util.h>
#include <tuple>
#include <iostream>
[[includes]]

namespace AOTRITON_NS::v3::[[family_name]] {

int64_t [[context_class_name]]::godel_number() const
{
    int64_t sum = 0;
    const auto& args = *params;
[[godel_number_body]]
    return sum;
}

hipError_t
[[context_class_name]]::lookup_optimal(Gpu gpu) {
    auto [arch_number, mod_number] = get_archmod_number(gpu);
    if (arch_number < 0) {
        return hipErrorNoBinaryForGpu;
    }
    backend_index = BackendEnum::None;
    auto number = godel_number();
    if (number < 0)
        return hipErrorNotSupported;
    auto tune_func = optune_table[arch_number][number];
    if (!tune_func)
        return hipErrorProfilerNotInitialized;
    tune_func(*this, mod_number);
    // In case tuning database is broken
    if (backend_index < 0)
        backend_index = fallback_backend;
    return hipSuccess;
}

std::tuple<int, int>
[[context_class_name]]::get_archmod_number(Gpu gpu) {
    [[get_archmod_number_body]];
    // TODO: print warning about tuning for this GPU mod is not built.
    // Note: if some mod does not have tuning info in the database at all, the
    //       getGpuFromStream should not return that mod from beginning.
    return std::make_tuple(-1, 0);
}

hipError_t
[[context_class_name]]::launch(Gpu gpu, hipStream_t stream) const {
    if (backend_index < 0) {
        return hipErrorPriorLaunchFailure;
    }
    auto ret = launcher_table[backend_index](*this, gpu, stream);
    // It is possible that the optimal backend does not support certain inputs
    // In this case hipErrorPeerAccessUnsupported will be returned
    if (ret == hipErrorPeerAccessUnsupported) {
        if (!disable_fallback) {
#ifndef NDEBUG
          std::cerr << "[[context_class_name]]::launch failed with optimal backend, "
                     << "calling fallback." << std::endl;
#endif
          return launcher_table[fallback_backend](*this, gpu, stream);
        }
#ifndef NDEBUG
        std::cerr << "[[context_class_name]]::launch failed with optimal backend, "
                   << "fallback disabled, returning error." << std::endl;
#endif
    }
    return ret;
}

// Launchers are defined in op source file and no need to put them in to
// optune namespace
namespace {
[[def_backend_launchers]]
}

[[context_class_name]]::BackendLauncher
[[context_class_name]]::launcher_table[ BackendEnum::Max ] = {
    [[launcher_table_entries]]
};

namespace optune {

[[list_of_deduplicated_lut_functions]]

} // namespace autotune

// When Functional's LUT is uniform or empty
namespace {
[[def_trivial_tunes]]
}

[[context_class_name]]::OpTuneTableEntry
[[context_class_name]]::optune_table[][ [[number_of_functionals]] ] = {
[[optune_table_entries]]
};

}

// vim: set fileencoding=utf-8
