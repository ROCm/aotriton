# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH CMAKE_CURRENT_SOURCE_PARENT_DIR)
message("CMAKE_CURRENT_SOURCE_PARENT_DIR ${CMAKE_CURRENT_SOURCE_PARENT_DIR}")
message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_V2_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_V2_BUILD_DIR}")

get_filename_component(AOTRITON_COMPILER "${CMAKE_CURRENT_LIST_DIR}/../v2python/compile.py" ABSOLUTE)
message("AOTRITON_COMPILER ${AOTRITON_COMPILER}")

# set(AOTRITON_GEN_FLAGS "")
# if(AOTRITON_COMPRESS_KERNEL)
#   list(APPEND AOTRITON_GEN_FLAGS "--enable_zstd" "${ZSTD_EXEC}")
# endif(AOTRITON_COMPRESS_KERNEL)

# add_custom_target(aotriton_v2_gen_compile
#   COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python -m v2python.generate_compile --target_gpus ${TARGET_GPUS} --build_dir "${AOTRITON_V2_BUILD_DIR}" ${AOTRITON_GEN_FLAGS}
#   WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
#   BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/Makefile.compile"
# )
# add_dependencies(aotriton_v2_gen_compile aotriton_venv_triton)

if(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "--build_for_tuning")
else(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "")
endif()
execute_process(
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python" -m v2python.generate_compile --target_gpus ${TARGET_GPUS} --build_dir "${AOTRITON_V2_BUILD_DIR}" --bare_mode ${GENERATE_OPTION}
  COMMAND_ECHO STDOUT
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
)
message("Bare.compile: ${AOTRITON_V2_BUILD_DIR}/Bare.compile")
file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.compile" HSACO_RULES)
set(ALL_HSACOS "")
foreach(RULE IN LISTS HSACO_RULES)
  # message("${RULE}")
  list(POP_FRONT RULE HSACO)
  list(POP_FRONT RULE SRC)
  list(POP_FRONT RULE KNAME)
  list(POP_FRONT RULE NWARPS)
  list(POP_FRONT RULE NSTAGES)
  list(POP_FRONT RULE WAVESPEREU)
  list(POP_FRONT RULE TGTGPU)
  list(POP_FRONT RULE SIG)
  if(AOTRITON_COMPRESS_KERNEL)
    add_custom_command(OUTPUT "${HSACO}.zst"
      COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python"
      "${AOTRITON_COMPILER}"
      "${SRC}"
      "--kernel_name" "${KNAME}"
      "-o" "${HSACO}"
      "-g" "1,1,1"
      "--num_warps" "${NWARPS}"
      "--num_stages" "${NSTAGES}"
      "--waves_per_eu" "${WAVESPEREU}"
      "--target" "${TGTGPU}"
      "--signature" "${SIG}"
      COMMAND ${ZSTD_EXEC} "-f" ${HSACO}
      DEPENDS aotriton_venv_triton
    )
    list(APPEND ALL_HSACOS "${HSACO}.zst")
  else(AOTRITON_COMPRESS_KERNEL)
    add_custom_command(OUTPUT "${HSACO}"
      COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python"
      "${AOTRITON_COMPILER}"
      "${SRC}"
      "--kernel_name" "${KNAME}"
      "-o" "${HSACO}"
      "-g" "1,1,1"
      "--num_warps" "${NWARPS}"
      "--num_stages" "${NSTAGES}"
      "--waves_per_eu" "${WAVESPEREU}"
      "--target" "${TGTGPU}"
      "--signature" "${SIG}"
      DEPENDS aotriton_venv_triton
    )
    list(APPEND ALL_HSACOS "${HSACO}")
  endif(AOTRITON_COMPRESS_KERNEL)
  # message("HSACO ${HSACO}")
endforeach(RULE)

if(DEFINED ENV{MAX_JOBS})
  set(MAX_JOBS "$ENV{MAX_JOBS}")
else()
  cmake_host_system_information(RESULT MAX_JOBS QUERY NUMBER_OF_PHYSICAL_CORES)
  if(MAX_JOBS LESS 2) # In case of failures.
    set(MAX_JOBS 2)
  endif()
endif()

add_custom_target(aotriton_v2_compile ALL DEPENDS ${ALL_HSACOS})

# add_custom_target(aotriton_v2_compile
#   # (CAVEAT) KNOWN PROBLEM: Will not work if LD_PRELOAD is not empty
#   # FIXME: Change this into `-E env --modify LD_PRELOAD=path_list_prepend:${AMDOCL_LD_PRELOAD}` when minimal cmake >= 3.25
#   COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" make -j ${MAX_JOBS} -f Makefile.compile LIBHSA_RUNTIME64=${AMDHSA_LD_PRELOAD}
#   WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}"
#   COMMAND_EXPAND_LISTS
#   BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/flash/attn_fwd.h"
#   "${AOTRITON_V2_BUILD_DIR}/flash/attn_fwd.cc"
#   # There are other by-products we did not bother to list here
# )

set(AOTRITON_SHIM_FLAGS "")
if(AOTRITON_NO_SHARED)
    list(APPEND AOTRITON_SHIM_FLAGS "--archive_only" "${AOTRITON_SHIM_FLAGS}")
endif(AOTRITON_NO_SHARED)
if(AOTRITON_ZSTD_INCLUDE)
    list(APPEND AOTRITON_SHIM_FLAGS "--enable_zstd" "${AOTRITON_ZSTD_INCLUDE}")
endif()
message(STATUS "AOTRITON_ZSTD_INCLUDE ${AOTRITON_ZSTD_INCLUDE}")
message(STATUS "AOTRITON_SHIM_FLAGS ${AOTRITON_SHIM_FLAGS}")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python" -m v2python.generate_shim --target_gpus ${TARGET_GPUS} --build_dir ${AOTRITON_V2_BUILD_DIR} --bare_mode "${GENERATE_OPTION}"
  COMMAND_ECHO STDOUT
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
)
file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.shim" SHIM_CC_FILES)

# CAVEAT: Actual shim code can only be generated during build phase because it
# requires some kernel information. (Notably shared memory requirement)
add_custom_target(aotriton_v2_gen_shim 
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python" -m v2python.generate_shim --target_gpus ${TARGET_GPUS} --build_dir ${AOTRITON_V2_BUILD_DIR} ${AOTRITON_SHIM_FLAGS} "${GENERATE_OPTION}"
  BYPRODUCTS ${SHIM_CC_FILES}  # Essential, otherwise add_library complains
  COMMAND_EXPAND_LISTS
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
)
add_dependencies(aotriton_v2_gen_shim aotriton_v2_compile)

if(AOTRITON_NO_SHARED)
  add_library(aotriton_v2 STATIC ${SHIM_CC_FILES})
else(AOTRITON_NO_SHARED)
  add_library(aotriton_v2 SHARED ${SHIM_CC_FILES})
endif(AOTRITON_NO_SHARED)
add_dependencies(aotriton_v2 aotriton_v2_gen_shim)

if(AOTRITON_ZSTD_INCLUDE)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_USE_ZSTD=1)
  target_include_directories(aotriton_v2 PRIVATE ${AOTRITON_ZSTD_INCLUDE})
else(AOTRITON_ZSTD_INCLUDE)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_USE_ZSTD=0)
endif(AOTRITON_ZSTD_INCLUDE)
target_include_directories(aotriton_v2 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
target_include_directories(aotriton_v2 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(aotriton_v2 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../third_party/incbin)
target_compile_options(aotriton_v2 PRIVATE -fPIC --no-offload-arch=all)
if(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
else(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
endif(AOTRITON_BUILD_FOR_TUNING)

# message(STATUS "AOTRITON_EXTRA_COMPILER_OPTIONS ${AOTRITON_EXTRA_COMPILER_OPTIONS}")
# add_custom_target(aotriton_v2
#   ALL
#   COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" make -j ${MAX_JOBS} -f Makefile.shim HIPCC=${AOTRITON_HIPCC_PATH} AR=${CMAKE_AR} EXTRA_COMPILER_OPTIONS=${AOTRITON_EXTRA_COMPILER_OPTIONS}
#   WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}"
#   BYPRODUCTS "${AOTRITON_V2_BUILD_DIR}/libaotriton_v2.a"
# )
# add_dependencies(aotriton_v2 aotriton_v2_gen_shim)

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}/include/aotriton" DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})
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
target_include_directories(aotriton INTERFACE ${CMAKE_CURRENT_SOURCE_PARENT_DIR}/include)
