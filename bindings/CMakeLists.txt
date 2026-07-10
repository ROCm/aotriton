# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

aux_source_directory(. PYAOTRITON_SRC)
if(AOTRITON_USE_TORCH)
  aux_source_directory(lazy_tensor PYAOTRITON_SRC)
endif(AOTRITON_USE_TORCH)
# Per-family API bindings: modules/<family>/bindings/*.cc self-register their
# submodules via SubmoduleRegistrar (submodule_registry.h). Convention over
# hardcoding — a new family drops in modules/<fam>/bindings, no edit here.
file(GLOB MODULE_BINDING_SRC CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/modules/*/bindings/*.cc")
list(APPEND PYAOTRITON_SRC ${MODULE_BINDING_SRC})
# find_package(hip REQUIRED)
pybind11_add_module(pyaotriton ${PYAOTRITON_SRC})
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

if(AOTRITON_USE_TORCH)
  target_compile_definitions(pyaotriton PRIVATE -DAOTRITON_USE_TORCH=1)
  target_link_libraries(pyaotriton PRIVATE torch)
else()
  target_compile_definitions(pyaotriton PRIVATE -DAOTRITON_USE_TORCH=0)
endif()

set_target_properties(pyaotriton PROPERTIES INSTALL_RPATH "$ORIGIN")
include(GNUInstallDirs)
install(TARGETS pyaotriton LIBRARY DESTINATION lib)
