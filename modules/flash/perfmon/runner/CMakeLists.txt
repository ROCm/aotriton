# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# One subject's flash harness: libperfmon_flash.so (the adapter of one AOTriton
# release) and bin/runner (perfmon/core/main.cc linked with it), built against
# that subject's own AOTriton install and the shared core of its ROCm.
# Configured per subject by perfmon/build_subject.sh; the core is configured once
# per ROCm by perfmon/build_core.sh.
#
#   -DPERFMON_FLASH_TAG=<tag>        selects runner/<tag>/adapter.cc
#   -DAOTRITON_ROOT=<dir>            the subject: include/aotriton/*.h and
#                                    lib/libaotriton*_v2.so of ONE build
#   -DPERFMON_CORE_ROOT=<dir>        installed/perfmon/core
#   -DPERFMON_ROCM_VERSION=<ver>     picks lib/libperfmon_core+rocm<ver>.so
#   -DCMAKE_INSTALL_PREFIX=<dir>     the subject, normally == AOTRITON_ROOT:
#                                    installed/perfmon/<arch>/aotriton-<tag>+rocm<ver>
#
# Installed layout (the whole installed/perfmon/ tree can be moved):
#   <subject>/bin/runner               RUNPATH $ORIGIN/../lib, $ORIGIN/../../../core/lib
#   <subject>/lib/libperfmon_flash.so  RUNPATH $ORIGIN, $ORIGIN/../../../core/lib
#   <subject>/lib/libaotriton*_v2.so
#   <core>/lib/libperfmon_core+rocm<ver>.so
# followed, in both, by the ROCm runtime packages (never the devel package).
# DT_RUNPATH is not transitive, so each binary carries its own entries.

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PerfmonFlashAdapter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PERFMON_FLASH_TAG "" CACHE STRING "AOTriton tag of this subject, e.g. 0.13b")
set(AOTRITON_ROOT "" CACHE PATH "Install prefix of the subject's AOTriton")
set(PERFMON_CORE_ROOT "" CACHE PATH "installed/perfmon/core")
set(PERFMON_ROCM_VERSION "" CACHE STRING "ROCm version of the core, e.g. 10.1.0")

get_filename_component(_perfmon_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
set(_perfmon_core_src "${_perfmon_repo_root}/perfmon/core")
# The flash entry codec: part of the family, compiled into libperfmon_flash
# (the core has no family knowledge). It reads the PON with the core's
# installed header-only include/perfmon/pon.h.
set(_perfmon_core_ext "${_perfmon_repo_root}/modules/flash/perfmon/core_ext")
if(NOT EXISTS "${_perfmon_core_src}/main.cc")
  message(FATAL_ERROR "[perfmon_flash] ${_perfmon_core_src}/main.cc not found: configure from the "
                      "checkout that holds perfmon/core.")
endif()

foreach(_var PERFMON_FLASH_TAG AOTRITON_ROOT PERFMON_CORE_ROOT PERFMON_ROCM_VERSION)
  if(NOT ${_var})
    message(FATAL_ERROR "[perfmon_flash] -D${_var}=... is required (see the top of this file).")
  endif()
endforeach()

# --- Adapter -------------------------------------------------------------------
# One directory per API generation. A tag whose API is the same is a symlink to
# it, so the two cannot drift apart unnoticed; when one diverges, the symlink
# becomes a directory. No adapter for a tag is an error rather than a fallback to
# a neighbour's: the API is not stable, and the compiler output of the wrong
# adapter says nothing about the real cause.
set(_perfmon_tag_dir "${CMAKE_CURRENT_SOURCE_DIR}/${PERFMON_FLASH_TAG}")
if(NOT EXISTS "${_perfmon_tag_dir}/adapter.cc")
  message(FATAL_ERROR "[perfmon_flash] no adapter for tag '${PERFMON_FLASH_TAG}': "
                      "${_perfmon_tag_dir}/adapter.cc does not exist. Write one against that "
                      "tag's include/aotriton/flash.h, or symlink the directory to a tag with "
                      "an identical API.")
endif()

# --- The subject's AOTriton --------------------------------------------------------
if(NOT EXISTS "${AOTRITON_ROOT}/include/aotriton/flash.h")
  message(FATAL_ERROR "[perfmon_flash] ${AOTRITON_ROOT}/include/aotriton/flash.h not found: "
                      "AOTRITON_ROOT is not an AOTriton install prefix.")
endif()
# Exactly one library: a second one would mean a stale or shared prefix, and a
# missing one would let the linker fall back to a system AOTriton.
file(GLOB _perfmon_aotriton_libs "${AOTRITON_ROOT}/lib/libaotriton*_v2.so"
                                 "${AOTRITON_ROOT}/lib64/libaotriton*_v2.so")
list(LENGTH _perfmon_aotriton_libs _perfmon_n_aotriton)
if(NOT _perfmon_n_aotriton EQUAL 1)
  message(FATAL_ERROR "[perfmon_flash] expected exactly one libaotriton*_v2.so under "
                      "${AOTRITON_ROOT}/lib, found ${_perfmon_n_aotriton}: ${_perfmon_aotriton_libs}")
endif()
list(GET _perfmon_aotriton_libs 0 _perfmon_aotriton_lib)

# The subject's own libraries are found through $ORIGIN, so AOTriton must be
# installed into the prefix this project installs into.
get_filename_component(_perfmon_aotriton_real "${AOTRITON_ROOT}" REALPATH)
get_filename_component(_perfmon_install_real "${CMAKE_INSTALL_PREFIX}" REALPATH)
if(NOT _perfmon_aotriton_real STREQUAL _perfmon_install_real)
  message(FATAL_ERROR "[perfmon_flash] AOTRITON_ROOT (${_perfmon_aotriton_real}) must be the install "
                      "prefix (${_perfmon_install_real}): the RUNPATH $ORIGIN/../lib finds "
                      "libaotriton there.")
endif()

# --- The core ------------------------------------------------------------------------
# The RUNPATHs below reach the core as <install>/../../core/lib, so the core must
# be where build_core.sh puts it relative to the subject.
get_filename_component(_perfmon_expected_core "${CMAKE_INSTALL_PREFIX}/../../core" REALPATH)
get_filename_component(_perfmon_given_core "${PERFMON_CORE_ROOT}" REALPATH)
if(NOT _perfmon_expected_core STREQUAL _perfmon_given_core)
  message(FATAL_ERROR "[perfmon_flash] PERFMON_CORE_ROOT must be <install>/../../core: install "
                      "prefix ${CMAKE_INSTALL_PREFIX} implies ${_perfmon_expected_core}, got "
                      "${_perfmon_given_core}. The RUNPATH $ORIGIN/../../../core/lib would not find it.")
endif()
set(_perfmon_core_lib "${PERFMON_CORE_ROOT}/lib/libperfmon_core+rocm${PERFMON_ROCM_VERSION}.so")
if(NOT EXISTS "${_perfmon_core_lib}" OR NOT EXISTS "${PERFMON_CORE_ROOT}/include/perfmon/perfmon_abi.h"
   OR NOT EXISTS "${PERFMON_CORE_ROOT}/include/perfmon/pon.h")
  message(FATAL_ERROR "[perfmon_flash] ${_perfmon_core_lib} or the core's include/perfmon/ is missing: "
                      "run perfmon/build_core.sh --rocm ${PERFMON_ROCM_VERSION} first.")
endif()

# --- Backend table -----------------------------------------------------------------------
include("${CMAKE_CURRENT_SOURCE_DIR}/GenerateBackends.cmake")
set(_perfmon_backends_csv "${_perfmon_tag_dir}/backends.csv")
set(_perfmon_backends_h "${AOTRITON_ROOT}/include/aotriton/flash/backends.h")
if(EXISTS "${_perfmon_backends_csv}")
  perfmon_flash_read_backends_csv("${_perfmon_backends_csv}" _perfmon_fwd_backends _perfmon_bwd_backends)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_perfmon_backends_csv}")
  set(_perfmon_backends_origin "${_perfmon_backends_csv}")
elseif(EXISTS "${_perfmon_backends_h}")
  perfmon_flash_read_backends_header("${_perfmon_backends_h}" _perfmon_fwd_backends _perfmon_bwd_backends)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_perfmon_backends_h}")
  set(_perfmon_backends_origin "${_perfmon_backends_h}")
else()
  message(FATAL_ERROR "[perfmon_flash] no backend table for '${PERFMON_FLASH_TAG}': neither "
                      "${_perfmon_backends_csv} nor ${_perfmon_backends_h} exists.")
endif()
perfmon_flash_write_backends_table("${CMAKE_CURRENT_BINARY_DIR}/backends_table.cc"
                                   _perfmon_fwd_backends _perfmon_bwd_backends)
message(STATUS "[perfmon_flash] backends from ${_perfmon_backends_origin}:")
message(STATUS "[perfmon_flash]   op_attn_fwd: ${_perfmon_fwd_backends}")
message(STATUS "[perfmon_flash]   op_attn_bwd: ${_perfmon_bwd_backends}")

# --- HIP and RUNPATHs ----------------------------------------------------------------------
# The same ROCm discovery as AOTriton's own build (ROCM_PATH, then rocm-sdk, then
# /opt/rocm), and the same runtime-package RUNPATHs as the core.
include("${_perfmon_repo_root}/cmake/LoadHIP.cmake")
include("${_perfmon_core_src}/PerfmonRocmRuntime.cmake")
perfmon_rocm_runtime_dirs(_perfmon_rocm_rpath "${ROCM_PATH}")

set(_perfmon_rpath_adapter "$ORIGIN" "$ORIGIN/../../../core/lib" ${_perfmon_rocm_rpath})
set(_perfmon_rpath_runner "$ORIGIN/../lib" "$ORIGIN/../../../core/lib" ${_perfmon_rocm_rpath})

# --- libperfmon_flash.so -----------------------------------------------------------------------
add_library(perfmon_flash SHARED
  "${_perfmon_tag_dir}/adapter.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/common.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/family.cc"
  "${_perfmon_core_ext}/entry_codec.cc"
  "${CMAKE_CURRENT_BINARY_DIR}/backends_table.cc")
target_include_directories(perfmon_flash PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/lib"  # for the generated backends_table.cc
  "${_perfmon_core_ext}"             # entry_codec.h; it includes <perfmon/pon.h> from the core
  "${AOTRITON_ROOT}/include"
  "${PERFMON_CORE_ROOT}/include")
target_link_libraries(perfmon_flash PRIVATE
  hip::host
  "${_perfmon_aotriton_lib}"
  "${_perfmon_core_lib}")
set_target_properties(perfmon_flash PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_perfmon_rpath_adapter}")
target_link_options(perfmon_flash PRIVATE "LINKER:--enable-new-dtags")

# --- bin/runner -------------------------------------------------------------------------------------
# main.cc comes from the checkout, never from a release's tree (old releases
# predate it). It passes its PMON_ABI_VERSION to the core, which refuses a
# mismatch at run time.
add_executable(runner "${_perfmon_core_src}/main.cc")
target_include_directories(runner PRIVATE "${PERFMON_CORE_ROOT}/include")
target_link_libraries(runner PRIVATE
  perfmon_flash
  hip::host
  "${_perfmon_core_lib}")
set_target_properties(runner PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_perfmon_rpath_runner}")
target_link_options(runner PRIVATE "LINKER:--enable-new-dtags")

install(TARGETS perfmon_flash LIBRARY DESTINATION lib)
install(TARGETS runner RUNTIME DESTINATION bin)
