# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# libperfmon_core+rocm<ver>.so: the measurement harness shared by every
# subject. Built once per ROCm, as one multi-arch binary, by
# perfmon/build_core.sh, and installed to installed/perfmon/core/{include,lib}.
#
# It never sees an AOTriton header (checked below). That is what makes timings
# of two AOTriton releases comparable: the harness cannot differ between them.
# For the same reason it is a standalone project, never add_subdirectory()'d
# from AOTriton's own build.
#
# Configure with CXX=hipcc: fill.cc holds the core's only kernel.
#
#   -DPERFMON_ROCM_VERSION=<ver>    required; part of the library's name
#   -DPERFMON_CORE_ARCHES=<a;b;..>  required; arches to compile the kernel for
#   -DPERFMON_CORE_COMMIT=<sha>     the source commit, reported by the runner
#   -DPERFMON_ENABLE_AMDSMI=ON      the thermal gate's temperature source

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PerfmonCore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PERFMON_ENABLE_AMDSMI "Use amd-smi for the thermal gate (OFF: thermal_stub.cc, the gate cannot be enabled)" OFF)
set(PERFMON_ROCM_VERSION "" CACHE STRING "ROCm version this core is built against, e.g. 10.1.0")
set(PERFMON_CORE_ARCHES "" CACHE STRING "Arches to build device code for, ';'-separated")
set(PERFMON_CORE_COMMIT "unknown" CACHE STRING "Source commit of this core")

if(NOT PERFMON_ROCM_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "[perfmon_core] PERFMON_ROCM_VERSION must be <major>.<minor>.<patch>, "
                      "got '${PERFMON_ROCM_VERSION}'")
endif()
if(NOT PERFMON_CORE_ARCHES)
  # Without an explicit list hipcc compiles for its own default, and a missing
  # arch only fails at run time ("no kernel image is available").
  message(FATAL_ERROR "[perfmon_core] PERFMON_CORE_ARCHES is required")
endif()

get_filename_component(_perfmon_repo_root "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# The same ROCm discovery as AOTriton's own build: ROCM_PATH, then rocm-sdk,
# then /opt/rocm. Sets ROCM_PATH and finds hip.
include("${_perfmon_repo_root}/cmake/LoadHIP.cmake")

include("${CMAKE_CURRENT_SOURCE_DIR}/PerfmonRocmRuntime.cmake")
perfmon_rocm_runtime_dirs(_perfmon_rocm_rpath "${ROCM_PATH}")

if(PERFMON_ENABLE_AMDSMI)
  set(_perfmon_thermal_src thermal_amdsmi.cc)
else()
  set(_perfmon_thermal_src thermal_stub.cc)
endif()

configure_file(core_info.cc.in "${CMAKE_CURRENT_BINARY_DIR}/core_info.cc" @ONLY)

add_library(perfmon_core SHARED
  abi.cc
  protocol.cc
  stats.cc
  config.cc
  fill.cc
  timing.cc
  runner.cc
  ${_perfmon_thermal_src}
  "${CMAKE_CURRENT_BINARY_DIR}/core_info.cc")

target_include_directories(perfmon_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(perfmon_core PUBLIC hip::host)

foreach(_arch IN LISTS PERFMON_CORE_ARCHES)
  target_compile_options(perfmon_core PRIVATE --offload-arch=${_arch})
endforeach()
message(STATUS "[perfmon_core] device code for: ${PERFMON_CORE_ARCHES}")

# The ROCm version is in the file name and the soname, so the cores of
# several ROCm versions share one lib/ directory. No VERSION/SOVERSION: a
# rebuild (e.g. for a longer arch list) keeps the name, and existing subjects
# need no relink.
set_target_properties(perfmon_core PROPERTIES
  OUTPUT_NAME "perfmon_core+rocm${PERFMON_ROCM_VERSION}"
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${_perfmon_rocm_rpath}")
target_link_options(perfmon_core PRIVATE "LINKER:--enable-new-dtags")

if(PERFMON_ENABLE_AMDSMI)
  # No soft fallback: a missing amd-smi must fail the build, not produce a
  # core whose gate silently does nothing.
  target_include_directories(perfmon_core PRIVATE "${ROCM_PATH}/include")
  target_link_libraries(perfmon_core PRIVATE amd_smi)
endif()

# The core must never see AOTriton headers. Tested by what an include
# directory contains, not by its name: a checkout named aotriton.src would
# otherwise trip a name match, and an install prefix with any name can hold
# aotriton/flash.h.
function(perfmon_assert_no_aotriton_includes target)
  get_target_property(_incs ${target} INCLUDE_DIRECTORIES)
  if(NOT _incs)
    return()
  endif()
  set(_forbidden_roots "${_perfmon_repo_root}/include" "${_perfmon_repo_root}/v3src")
  foreach(_inc IN LISTS _incs)
    get_filename_component(_inc_real "${_inc}" REALPATH)
    if(EXISTS "${_inc_real}/aotriton/config.h" OR EXISTS "${_inc_real}/aotriton/flash.h")
      message(FATAL_ERROR "[perfmon_core] ${target} can see AOTriton headers via ${_inc}")
    endif()
    foreach(_root IN LISTS _forbidden_roots)
      if(_inc_real STREQUAL "${_root}" OR _inc_real MATCHES "^${_root}/")
        message(FATAL_ERROR "[perfmon_core] ${target} includes ${_inc}, inside AOTriton's ${_root}")
      endif()
    endforeach()
  endforeach()
endfunction()
perfmon_assert_no_aotriton_includes(perfmon_core)

# Fixed lib/, not CMAKE_INSTALL_LIBDIR: subjects reach the core through the
# RUNPATH $ORIGIN/../../../core/lib.
install(TARGETS perfmon_core LIBRARY DESTINATION lib)
# What adapters and main.cc include.
install(FILES perfmon_abi.h fill.h pon.h DESTINATION include/perfmon)
