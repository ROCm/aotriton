# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# CPU-only unit tests of the flash entry codec. Plain host compiler, no HIP, no
# AOTriton; the PON reader is the central core's header-only pon.h.
#
#   cmake -S modules/flash/perfmon/core_ext/tests -B /tmp/perfmon-flash-tests -G Ninja
#   cmake --build /tmp/perfmon-flash-tests && ctest --test-dir /tmp/perfmon-flash-tests

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(PerfmonFlashCoreExtTests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(_core_ext "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
get_filename_component(_repo "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)
set(_core "${_repo}/perfmon/core")

enable_testing()
add_executable(test_entry_codec test_entry_codec.cc "${_core_ext}/entry_codec.cc")
# entry_codec.cc includes <perfmon/pon.h>, as it does in an adapter build; here
# that is perfmon/core/ through a one-file directory, so the layout matches.
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/perfmon")
configure_file("${_core}/pon.h" "${CMAKE_CURRENT_BINARY_DIR}/include/perfmon/pon.h" COPYONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_core}/pon.h")
target_include_directories(test_entry_codec PRIVATE
  "${_core_ext}"
  "${_repo}/perfmon/core/tests"  # check.h
  "${CMAKE_CURRENT_BINARY_DIR}/include")
target_compile_options(test_entry_codec PRIVATE -Wall -Wextra -Werror)
add_test(NAME entry_codec COMMAND test_entry_codec)
