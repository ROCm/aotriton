# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_BUILD_DIR}")

add_custom_target(aotriton_v1_gen_compile
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python python/generate.py --target ${TARGET_GPUS} --build_dir "${AOTRITON_BUILD_DIR}"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  BYPRODUCTS "${AOTRITON_BUILD_DIR}/Makefile.compile"
)
add_dependencies(aotriton_v1_gen_compile aotriton_venv_triton)

include(ProcessorCount)
ProcessorCount(PROCS)
add_custom_target(aotriton_v1_compile
  # (CAVEAT) KNOWN PROBLEM: Will not work if LD_PRELOAD is not empty
  # FIXME: Change this into `-E env --modify LD_PRELOAD=path_list_prepend:${AMDOCL_LD_PRELOAD}` when minimal cmake >= 3.25
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" make -f Makefile.compile LIBHSA_RUNTIME64=${AMDHSA_LD_PRELOAD}
  WORKING_DIRECTORY "${AOTRITON_BUILD_DIR}"
  BYPRODUCTS "${AOTRITON_BUILD_DIR}/attn_fwd.h"
  "${AOTRITON_BUILD_DIR}/bwd_kernel_dk_dv.h"
  "${AOTRITON_BUILD_DIR}/bwd_kernel_dq.h"
  "${AOTRITON_BUILD_DIR}/bwd_preprocess.h"
)
add_dependencies(aotriton_v1_compile aotriton_v1_gen_compile)

add_custom_target(aotriton_v1_gen_shim
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python python/generate_shim.py --build_dir "${AOTRITON_BUILD_DIR}" --archive
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  BYPRODUCTS "${AOTRITON_BUILD_DIR}/Makefile.shim"
)
add_dependencies(aotriton_v1_gen_shim aotriton_v1_compile) # Shim source files need json metadata

add_custom_target(aotriton_v1
  ALL
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" make -f Makefile.shim HIPCC=${AOTRITON_HIPCC_PATH} AR=${CMAKE_AR}
  WORKING_DIRECTORY "${AOTRITON_BUILD_DIR}"
  BYPRODUCTS "${AOTRITON_BUILD_DIR}/libaotriton_v1.a"
)
add_dependencies(aotriton_v1 aotriton_v1_gen_shim)

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${AOTRITON_BUILD_DIR}/attn_fwd.h"
  "${AOTRITON_BUILD_DIR}/bwd_kernel_dk_dv.h"
  "${AOTRITON_BUILD_DIR}/bwd_kernel_dq.h"
  "${AOTRITON_BUILD_DIR}/bwd_preprocess.h"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aotriton)
install(FILES "${AOTRITON_BUILD_DIR}/libaotriton_v1.a" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
