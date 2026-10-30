# Copyright © 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Shared by perfmon/core/CMakeLists.txt and
# modules/flash/perfmon/runner/CMakeLists.txt, which both need the same
# answer to one question: which directories must be on an installed binary's
# RUNPATH for it to find ROCm's RUNTIME libraries?
#
# It is deliberately NOT "${ROCM_PATH}/lib".
#
# On theRock, ROCM_PATH points at the `_rocm_sdk_devel` Python package --
# headers, CMake config files, LLVM: everything needed to BUILD. The runtime
# libraries live in sibling packages:
#
#   _rocm_sdk_core       libamdhip64, libamd_comgr, libhsa-runtime64,
#                        libamd_smi, librocm_kpack, librocprofiler-register,
#                        and the private rocm_sysdeps/ + host-math/ copies
#   _rocm_sdk_libraries  the math libraries (rocblas, hipblaslt, ...)
#
# `_rocm_sdk_devel` also contains HARDLINKED copies of core's libraries --
# `libamdhip64.so.7` has a link count of 3 across the two packages. That is
# precisely what makes the bug invisible: RPATH entries pointing into devel
# resolve perfectly on a build host, and then fail on a deployment host that
# installed only core+libraries, where those paths do not exist at all. A
# perfmon runner is built once and shipped to measurement nodes, so that is
# the normal case, not an edge case.
#
# Classical ROCm (/opt/rocm) has no such split: build and runtime trees are
# the same one, and the fallback below handles it unchanged.

# perfmon_rocm_runtime_dirs(<out_var> <rocm_path>)
#
# Sets <out_var> in the caller's scope to a list of runtime library
# directories. Override wholesale with -DPERFMON_ROCM_RUNTIME_DIRS=... for a
# layout this does not recognize.
function(perfmon_rocm_runtime_dirs out_var rocm_path)
  if(PERFMON_ROCM_RUNTIME_DIRS)
    set(${out_var} "${PERFMON_ROCM_RUNTIME_DIRS}" PARENT_SCOPE)
    return()
  endif()

  get_filename_component(_pkg_name "${rocm_path}" NAME)
  get_filename_component(_pkg_parent "${rocm_path}" DIRECTORY)

  set(_roots "")
  if(_pkg_name STREQUAL "_rocm_sdk_devel")
    # theRock: resolve the runtime siblings and do NOT keep devel itself.
    # Leaving devel on the RPATH as a "harmless fallback" would silently
    # re-hide the very failure this function exists to surface, since the
    # hardlinked copies would satisfy every lookup on the build host.
    foreach(_sibling _rocm_sdk_core _rocm_sdk_libraries)
      if(IS_DIRECTORY "${_pkg_parent}/${_sibling}/lib")
        list(APPEND _roots "${_pkg_parent}/${_sibling}")
      endif()
    endforeach()
    if(NOT _roots)
      message(WARNING
        "[perfmon] ROCM_PATH looks like theRock's _rocm_sdk_devel "
        "(${rocm_path}) but neither _rocm_sdk_core nor _rocm_sdk_libraries "
        "is installed beside it. Falling back to baking devel paths into "
        "RPATH; the resulting binaries will not run on a host that has only "
        "the runtime packages installed.")
    endif()
  endif()

  if(NOT _roots)
    set(_roots "${rocm_path}")
  endif()

  set(_dirs "")
  foreach(_root IN LISTS _roots)
    list(APPEND _dirs "${_root}/lib")
    # theRock ships some dependencies as private, renamed copies that RPATH
    # does not otherwise resolve -- an acknowledged upstream packaging bug
    # whose sanctioned workaround is adding these to the search path.
    # .ci/common-vars.sh's add_rocm_sdk_ldconfig() adds exactly these two to
    # LD_LIBRARY_PATH for the same reason; putting them on the RPATH is that
    # workaround made permanent, so the runner needs no environment set up
    # around it (no LD_LIBRARY_PATH).
    foreach(_sysdep rocm_sysdeps host-math)
      if(IS_DIRECTORY "${_root}/lib/${_sysdep}/lib")
        list(APPEND _dirs "${_root}/lib/${_sysdep}/lib")
      endif()
    endforeach()
  endforeach()

  set(${out_var} "${_dirs}" PARENT_SCOPE)
endfunction()
