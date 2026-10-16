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
  # A theRock ROCM_PATH points into the Python environment and only has content
  # after `rocm-sdk path --root` has unpacked rocm[devel] there. A missing or
  # empty tree is the symptom of forgetting that step.
  if(NOT EXISTS "${ROCM_PATH}/lib/cmake/hip")
    message(FATAL_ERROR
      "[AOTriton] ROCM_PATH environment variable is set to ${ROCM_PATH}, but "
      "it does not contain a ROCm installation (${ROCM_PATH}/lib/cmake/hip is "
      "missing).\n"
      "If ROCM_PATH came from theRock, run `rocm-sdk path --root` once in a "
      "writable environment to unpack rocm[devel], then retry. Otherwise set "
      "ROCM_PATH to a valid ROCm tree, or unset it to let AOTriton query "
      "`rocm-sdk`.")
  endif()
  message(STATUS "[AOTriton] Using ROCM_PATH from environment: ${ROCM_PATH}")
  set(_aotriton_rocm_cmake_prefix "${ROCM_PATH}/lib/cmake")
else()
  find_program(ROCM_SDK_EXECUTABLE rocm-sdk)
  if(ROCM_SDK_EXECUTABLE)
    # A failing rocm-sdk (e.g. rocm-sdk-core installed without rocm[devel]) is
    # fatal: falling back to /opt/rocm would silently pick a different ROCm.
    execute_process(
      COMMAND "${ROCM_SDK_EXECUTABLE}" path --root
      OUTPUT_VARIABLE _rocm_sdk_root
      ERROR_VARIABLE _rocm_sdk_root_error
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _rocm_sdk_root_result)
    if(NOT _rocm_sdk_root_result EQUAL 0)
      set(_rocm_sdk_hint "")
      if(_rocm_sdk_root_error MATCHES "Read-only file system")
        # Common in containers: rocm-sdk unpacks _rocm_sdk_devel/ into the
        # (read-only) venv on first use.
        string(CONCAT _rocm_sdk_hint
          "\n\nrocm-sdk must unpack its devel files on first use but the Python "
          "environment is read-only. Either run `rocm-sdk path --root` once with "
          "a writable environment (the result persists), or set the ROCM_PATH "
          "environment variable to an already-unpacked ROCm tree.")
      endif()
      message(FATAL_ERROR
        "[AOTriton] `${ROCM_SDK_EXECUTABLE} path --root` failed "
        "(exit status ${_rocm_sdk_root_result}):\n"
        "${_rocm_sdk_root_error}${_rocm_sdk_root}${_rocm_sdk_hint}")
    endif()
    execute_process(
      COMMAND "${ROCM_SDK_EXECUTABLE}" path --cmake
      OUTPUT_VARIABLE _rocm_sdk_cmake
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _rocm_sdk_cmake_result
      ERROR_QUIET)
    if(EXISTS "${_rocm_sdk_root}")
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

find_package(hip REQUIRED CONFIG)
