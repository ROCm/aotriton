// Copyright © 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "submodule_registry.h"
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace pyaotriton {

namespace {
struct Entry {
  std::string name;
  std::string doc;
  SubmoduleSetup fn;
};

// Function-local static so registration during other TUs' static init is safe
// regardless of initialization order.
std::vector<Entry>& registry() {
  static std::vector<Entry> r;
  return r;
}
} // namespace

SubmoduleRegistrar::SubmoduleRegistrar(const char* name, const char* doc, SubmoduleSetup fn) {
  registry().emplace_back(name, doc ? doc : "", std::move(fn));
}

void setup_registered_submodules(py::module_& m) {
  for (auto& e : registry()) {
    // def_submodule is get-or-create: if two TUs register the same name, they
    // contribute to the same submodule instead of clobbering it.
    py::module_ sub = m.def_submodule(e.name.c_str(), e.doc.c_str());
    e.fn(sub);
  }
}

} // namespace pyaotriton
