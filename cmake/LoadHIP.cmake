# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Locate the ROCm installation, extend CMAKE_PREFIX_PATH, and find_package(hip).
# Modeled on pytorch's cmake/public/LoadHIP.cmake. Resolution order:
#   1. ROCM_PATH environment variable (must exist, otherwise fatal)
#   2. `rocm-sdk path --root` / `rocm-sdk path --cmake` (theRock Python SDK)
#   3. /opt/rocm (C:/opt/rocm on Windows)
#
# Effects: sets ROCM_PATH (normal variable) and the ROCM_PATH environment
# variable (hip-config-amd.cmake needs it to find AMDDeviceLibs), and appends
# to CMAKE_PREFIX_PATH, then calls find_package(hip REQUIRED).

set(_aotriton_rocm_cmake_prefix "")
if(DEFINED ENV{ROCM_PATH})
  file(TO_CMAKE_PATH "$ENV{ROCM_PATH}" ROCM_PATH)
  if(NOT EXISTS "${ROCM_PATH}")
    message(FATAL_ERROR
      "[AOTriton] ROCM_PATH environment variable is set to ${ROCM_PATH} but does not exist.\n"
      "Set a valid ROCM_PATH or unset it to let AOTriton query `rocm-sdk`.")
  endif()
  message(STATUS "[AOTriton] Using ROCM_PATH from environment: ${ROCM_PATH}")
  set(_aotriton_rocm_cmake_prefix "${ROCM_PATH}/lib/cmake")
else()
  find_program(ROCM_SDK_EXECUTABLE rocm-sdk)
  if(ROCM_SDK_EXECUTABLE)
    execute_process(
      COMMAND "${ROCM_SDK_EXECUTABLE}" path --root
      OUTPUT_VARIABLE _rocm_sdk_root
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _rocm_sdk_root_result
      ERROR_QUIET)
    execute_process(
      COMMAND "${ROCM_SDK_EXECUTABLE}" path --cmake
      OUTPUT_VARIABLE _rocm_sdk_cmake
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _rocm_sdk_cmake_result
      ERROR_QUIET)
    if(_rocm_sdk_root_result EQUAL 0 AND EXISTS "${_rocm_sdk_root}")
      file(TO_CMAKE_PATH "${_rocm_sdk_root}" ROCM_PATH)
      message(STATUS "[AOTriton] Found ROCm via rocm-sdk at: ${ROCM_PATH}")
      if(_rocm_sdk_cmake_result EQUAL 0 AND EXISTS "${_rocm_sdk_cmake}")
        file(TO_CMAKE_PATH "${_rocm_sdk_cmake}" _aotriton_rocm_cmake_prefix)
      else()
        set(_aotriton_rocm_cmake_prefix "${ROCM_PATH}/lib/cmake")
      endif()
    endif()
  endif()

  if(NOT ROCM_PATH)
    if(WIN32)
      set(ROCM_PATH "C:/opt/rocm")
    else()
      set(ROCM_PATH "/opt/rocm")
    endif()
    message(WARNING "[AOTriton] ROCM_PATH is not set and `rocm-sdk` is unavailable, "
                    "falling back to ${ROCM_PATH}. Set ROCM_PATH to select a specific "
                    "ROCm/theRock installation.")
    set(_aotriton_rocm_cmake_prefix "${ROCM_PATH}")
  endif()
endif()

list(APPEND CMAKE_PREFIX_PATH "${_aotriton_rocm_cmake_prefix}")
unset(_aotriton_rocm_cmake_prefix)

find_package(hip REQUIRED)
