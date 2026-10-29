# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# CPU-only unit tests for the parts of perfmon/core that need no HIP: the line
# protocol, statistics, the generic PON reader and the startup checks. Built with the
# host compiler; no ROCm needed.
#
#   cmake -S perfmon/core/tests -B /tmp/perfmon-tests -G Ninja
#   cmake --build /tmp/perfmon-tests && ctest --test-dir /tmp/perfmon-tests

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PerfmonCoreTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_core "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)

add_library(perfmon_core_cpu STATIC
  "${_core}/protocol.cc"
  "${_core}/stats.cc"
  "${_core}/config.cc")
target_include_directories(perfmon_core_cpu PUBLIC "${_core}")
target_compile_options(perfmon_core_cpu PRIVATE -Wall -Wextra -Werror)

enable_testing()
foreach(_t protocol stats pon config)
  add_executable(test_${_t} test_${_t}.cc)
  target_link_libraries(test_${_t} PRIVATE perfmon_core_cpu)
  target_compile_options(test_${_t} PRIVATE -Wall -Wextra -Werror)
  add_test(NAME ${_t} COMMAND test_${_t})
endforeach()
