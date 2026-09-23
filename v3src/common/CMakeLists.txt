# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Sources here must not call a real hip*() runtime function. HIP headers, for
# hipError_t/hipStream_t as types, are fine -- see hip::host below.
set(COMMON_CC_FILES
  cpp_tune.cc
  fd_unix.cc
  fd_win32.cc
  log.cc
  lszip.cc
  packed_kernel_common.cc
  pon.cc
)
# Assertion only: a .cc added here without a matching COMMON_CC_FILES entry
# fails the configure step instead of silently never being built.
aux_source_directory(. _COMMON_CC_FILES_DISCOVERED)
list(TRANSFORM _COMMON_CC_FILES_DISCOVERED REPLACE "^\\./" "")
list(SORT _COMMON_CC_FILES_DISCOVERED)
set(_COMMON_CC_FILES_EXPECTED ${COMMON_CC_FILES})
list(SORT _COMMON_CC_FILES_EXPECTED)
if(NOT _COMMON_CC_FILES_DISCOVERED STREQUAL _COMMON_CC_FILES_EXPECTED)
  message(FATAL_ERROR
    "v3src/common/*.cc source set changed: aux_source_directory discovered "
    "[${_COMMON_CC_FILES_DISCOVERED}] but v3src/common/CMakeLists.txt expects "
    "[${_COMMON_CC_FILES_EXPECTED}]. A source that calls a real hip*() runtime "
    "function belongs in v3src/hip-dependent/ instead.")
endif()

file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.shim" SHIM_CC_FILES ENCODING UTF-8)

# Modular family C++ sources: every modules/<family>/csrc/**.cc (recursively, so
# per-family subdirs like csrc/aiter are picked up). Convention over hardcoding —
# adding a family needs no edit here. CONFIGURE_DEPENDS re-globs when files change.
file(GLOB_RECURSE MODULE_CC_FILES CONFIGURE_DEPENDS
     "${PROJECT_SOURCE_DIR}/modules/*/csrc/*.cc")

# Static so one build (AOTRITON_COMMON_LIBRARY_ONLY_MODE=ON) can be reused unmodified as
# AOTRITON_COMMON_LIBRARY across every ROCm version in the release matrix.
if(AOTRITON_COMMON_LIBRARY)
  # Reuse a previously-built archive: skip compiling the sources entirely.
  add_library(aotriton_common STATIC IMPORTED GLOBAL)
  set_target_properties(aotriton_common PROPERTIES
    IMPORTED_LOCATION "${AOTRITON_COMMON_LIBRARY}")
else()
  add_library(aotriton_common STATIC ${SHIM_CC_FILES} ${COMMON_CC_FILES} ${MODULE_CC_FILES})
  set_target_properties(aotriton_common PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(aotriton_common PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  # for <build>/include/aotriton/config.h. Code should use <aotriton/config.h>
  target_include_directories(aotriton_common PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
  target_include_directories(aotriton_common PRIVATE ${AOTRITON_V2_BUILD_DIR})
  # for <build>/v3src/include/aotriton/<family>/backends.h -- the per-operator
  # backend-index constants aotriton.generate emits. Public because the index is:
  # attn_options::force_backend_index takes it.
  target_include_directories(aotriton_common PUBLIC $<BUILD_INTERFACE:${AOTRITON_V2_BUILD_DIR}/include>)
  set_target_properties(aotriton_common PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
  )
  if(AOTRITON_BUILD_FOR_TUNING)
    target_compile_definitions(aotriton_common PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
    add_dependencies(aotriton_common aotriton_v2_regen_shim)
  else(AOTRITON_BUILD_FOR_TUNING)
    target_compile_definitions(aotriton_common PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
  endif(AOTRITON_BUILD_FOR_TUNING)
  target_link_libraries(aotriton_common PRIVATE lzma_interface)
  # PUBLIC only to propagate HIP's include dir; a static lib links nothing until
  # aotriton_v2 (or a downstream consumer) pulls the archive in.
  target_link_libraries(aotriton_common PUBLIC hip::host)
endif()

if(AOTRITON_COMMON_LIBRARY_ONLY_MODE)
  # Static lib + headers only: no aotriton_v2, no GPU images (forced off via
  # AOTRITON_NOIMAGE_MODE, which AOTRITON_COMMON_LIBRARY_ONLY_MODE implies).
  include(GNUInstallDirs)
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/aotriton"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.h")
  install(DIRECTORY "${AOTRITON_V2_BUILD_DIR}/include/aotriton"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
          FILES_MATCHING PATTERN "*.h")
  install(TARGETS aotriton_common
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
