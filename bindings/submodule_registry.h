// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Self-registration hook for per-family pyaotriton submodules.
//
// The core module skeleton (module.cc) no longer hardcodes which family
// submodules exist. Instead, each family's binding translation unit (e.g.
// modules/flash/bindings/v2.cc) registers its top-level submodule via a static
// SubmoduleRegistrar, and module.cc calls setup_registered_submodules() to
// instantiate them all. Adding a family's bindings needs no edit to the core.
//
// All binding TUs are compiled directly into the pyaotriton module, so the
// static registrars run at module load, before PYBIND11_MODULE's setup body.

#pragma once

#include <pybind11/pybind11.h>
#include <functional>

namespace pyaotriton {

using SubmoduleSetup = std::function<void(pybind11::module_&)>;

// Registering the same submodule name from multiple TUs is allowed: the
// submodule is get-or-created, so several families may share one version
// namespace (e.g. both flash and a future family under "v2").
struct SubmoduleRegistrar {
  SubmoduleRegistrar(const char* name, const char* doc, SubmoduleSetup fn);
};

// Instantiate every registered submodule under m and run its setup.
void setup_registered_submodules(pybind11::module_& m);

} // namespace pyaotriton
