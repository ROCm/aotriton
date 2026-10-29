# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

aux_source_directory(. PYAOTRITON_SRC)
aux_source_directory(lazy_tensor PYAOTRITON_SRC)
# Per-family API bindings: modules/<family>/bindings/*.cc self-register their
# submodules via SubmoduleRegistrar (submodule_registry.h). Convention over
# hardcoding — a new family drops in modules/<fam>/bindings, no edit here.
file(GLOB MODULE_BINDING_SRC CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/modules/*/bindings/*.cc")
list(APPEND PYAOTRITON_SRC ${MODULE_BINDING_SRC})
# find_package(hip REQUIRED)
# NB_DOMAIN gives this module its own nanobind type registry. Two pyaotriton
# builds with different AOTRITON_NAME_SUFFIX bind the same HIP types
# (hipError_t, ...), and in a shared registry the second one aborts the
# interpreter on import. This replaces pybind11's py::module_local().
set(PYAOTRITON_NB_OPTIONS NB_DOMAIN "pyaotriton${AOTRITON_NAME_SUFFIX}")
if(PYAOTRITON_ABI3)
  # Split mode: the extension only uses the stable ABI and finds nanobind's
  # runtime in the nanobind_backend Python package.
  list(APPEND PYAOTRITON_NB_OPTIONS
       BACKEND_MODULE nanobind_backend STABLE_ABI_VERSION 3.12)
endif()
nanobind_add_module(pyaotriton ${PYAOTRITON_NB_OPTIONS} ${PYAOTRITON_SRC})
# Family binding TUs #include "submodule_registry.h" from the core bindings dir.
target_include_directories(pyaotriton PRIVATE ${CMAKE_CURRENT_LIST_DIR})
# target_link_libraries(pyaotriton PRIVATE hip::device)
target_compile_features(pyaotriton PRIVATE cxx_std_20)
target_link_libraries(pyaotriton PUBLIC aotriton)
if(AOTRITON_COMPRESS_KERNEL)
  target_link_libraries(pyaotriton PRIVATE zstd_interface)
endif()
# TODO: Unify build option marcos with "interface target+public compile definitions"
if(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(pyaotriton PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
else(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(pyaotriton PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
endif(AOTRITON_BUILD_FOR_TUNING)

# lazy_tensor calls libtorch through the stable ABI (lazy_tensor/torch_stable_min.h),
# so building needs no PyTorch. The calls are resolved at load time from the
# libtorch_cpu that `import torch` has loaded; this records that dependency by
# linking an empty stand-in whose SONAME is libtorch_cpu.so. The file itself is
# named differently, because CMake puts its directory in the build-tree RUNPATH
# and a file that is really called libtorch_cpu.so there would be loaded when
# torch has not been imported, which fails with a confusing undefined-symbol
# error instead of "libtorch_cpu.so: cannot open shared object file". It is
# never installed.
add_library(torch_cpu_stub SHARED torch_stub/libtorch_cpu_stub.cc)
set_target_properties(torch_cpu_stub PROPERTIES
  NO_SONAME ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/torch_stub")
target_link_options(torch_cpu_stub PRIVATE -Wl,-soname,libtorch_cpu.so)
target_link_libraries(pyaotriton PRIVATE
  -Wl,--push-state,--no-as-needed torch_cpu_stub -Wl,--pop-state)

set_target_properties(pyaotriton PROPERTIES INSTALL_RPATH "$ORIGIN")
include(GNUInstallDirs)
install(TARGETS pyaotriton LIBRARY DESTINATION lib)
