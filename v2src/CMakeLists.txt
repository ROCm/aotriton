# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_V2_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_V2_BUILD_DIR}")

set(AOTRITON_GEN_FLAGS "")
if(AOTRITON_COMPRESS_KERNEL)
  list(APPEND AOTRITON_GEN_FLAGS "--enable_zstd" "${ZSTD_EXEC}")
endif(AOTRITON_COMPRESS_KERNEL)
add_custom_target(aotriton_v2_gen_compile
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python -m v2python.generate_compile --target_gpus ${TARGET_GPUS} --build_dir "${AOTRITON_V2_BUILD_DIR}" ${AOTRITON_GEN_FLAGS}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/Makefile.compile"
)
add_dependencies(aotriton_v2_gen_compile aotriton_venv_triton)

if(DEFINED ENV{MAX_JOBS})
  set(MAX_JOBS "$ENV{MAX_JOBS}")
else()
  cmake_host_system_information(RESULT MAX_JOBS QUERY NUMBER_OF_PHYSICAL_CORES)
  if(MAX_JOBS LESS 2) # In case of failures.
    set(MAX_JOBS 2)
  endif()
endif()

add_custom_target(aotriton_v2_compile
  # (CAVEAT) KNOWN PROBLEM: Will not work if LD_PRELOAD is not empty
  # FIXME: Change this into `-E env --modify LD_PRELOAD=path_list_prepend:${AMDOCL_LD_PRELOAD}` when minimal cmake >= 3.25
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" make -j ${MAX_JOBS} -f Makefile.compile LIBHSA_RUNTIME64=${AMDHSA_LD_PRELOAD}
  WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}"
  COMMAND_EXPAND_LISTS
  BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/flash/attn_fwd.h"
  "${AOTRITON_V2_BUILD_DIR}/flash/attn_fwd.cc"
  # There are other by-products we did not bother to list here
)
add_dependencies(aotriton_v2_compile aotriton_v2_gen_compile)

set(AOTRITON_SHIM_FLAGS "")
if(AOTRITON_NO_SHARED)
    list(APPEND AOTRITON_SHIM_FLAGS "--archive_only" "${AOTRITON_SHIM_FLAGS}")
endif(AOTRITON_NO_SHARED)
if(AOTRITON_ZSTD_INCLUDE)
    list(APPEND AOTRITON_SHIM_FLAGS "--enable_zstd" "${AOTRITON_ZSTD_INCLUDE}")
endif()
message(STATUS "AOTRITON_ZSTD_INCLUDE ${AOTRITON_ZSTD_INCLUDE}")
message(STATUS "AOTRITON_SHIM_FLAGS ${AOTRITON_SHIM_FLAGS}")

add_custom_target(aotriton_v2_gen_shim
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python -m v2python.generate_shim --target_gpus ${TARGET_GPUS} --build_dir ${AOTRITON_V2_BUILD_DIR} ${AOTRITON_SHIM_FLAGS}
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  COMMAND_EXPAND_LISTS
  BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/Makefile.shim"
)
add_dependencies(aotriton_v2_gen_shim aotriton_v2_compile) # Shim source files need json metadata

message(STATUS "AOTRITON_EXTRA_COMPILER_OPTIONS ${AOTRITON_EXTRA_COMPILER_OPTIONS}")
add_custom_target(aotriton_v2
  ALL
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" make -j ${MAX_JOBS} -f Makefile.shim HIPCC=${AOTRITON_HIPCC_PATH} AR=${CMAKE_AR} EXTRA_COMPILER_OPTIONS=${AOTRITON_EXTRA_COMPILER_OPTIONS}
  WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}"
  BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/libaotriton_v2.a"
)
add_dependencies(aotriton_v2 aotriton_v2_gen_shim)

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/aotriton" DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
# A few considerations
# 1. .so file is used for sanity check to ensure all symbols are resolved,
#    but it will not be installed because .a is what you should use to avoid setting
#    search paths for shared libraries
# 2. The archive library will be installed into the hardcode lib/ directory, to avoid Debian vs RHEL divergence.
install(FILES "${AOTRITON_V2_BUILD_DIR}/libaotriton_v2.a" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# Python binding only available for AOTriton V2 API
add_library(aotriton INTERFACE)
add_dependencies(aotriton aotriton_v2)
# target_link_libraries(aotriton INTERFACE ${CMAKE_INSTALL_PREFIX}/lib/libaotriton_v2.a)
# target_include_directories(aotriton INTERFACE ${CMAKE_INSTALL_PREFIX}/include)
target_link_libraries(aotriton INTERFACE ${AOTRITON_V2_BUILD_DIR}/libaotriton_v2.a)
target_include_directories(aotriton INTERFACE ${CMAKE_SOURCE_DIR}/include)
