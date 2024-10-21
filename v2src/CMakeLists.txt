# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

message("CMAKE_SOURCE_DIR ${CMAKE_SOURCE_DIR}")
message("CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}")
cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH CMAKE_CURRENT_SOURCE_PARENT_DIR)
message("CMAKE_CURRENT_SOURCE_PARENT_DIR ${CMAKE_CURRENT_SOURCE_PARENT_DIR}")
message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_V2_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(AOTRITON_KERNEL_STORAGE_V2_DIR "${AOTRITON_V2_BUILD_DIR}/aotriton.images")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_V2_BUILD_DIR}")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_KERNEL_STORAGE_V2_DIR}")

get_filename_component(AOTRITON_COMPILER "${CMAKE_CURRENT_LIST_DIR}/../v2python/compile.py" ABSOLUTE)
message("AOTRITON_COMPILER ${AOTRITON_COMPILER}")

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

if(AOTRITON_ENABLE_FP32_INPUTS)
  set(AOTRITON_ENABLE_FP32 1)
else()
  set(AOTRITON_ENABLE_FP32 0)
endif()

# Compile HSACO Kernels

## Generate Bare.compile file
execute_process(
  COMMAND ${CMAKE_COMMAND}
  -E env VIRTUAL_ENV=${VENV_DIR}
  AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
  "${VENV_DIR}/bin/python" -m v2python.generate_compile
  --target_gpus ${TARGET_GPUS} 
  --build_dir "${AOTRITON_V2_BUILD_DIR}"
  --bare_mode ${GENERATE_OPTION}
  --generate_cluster_info
  COMMAND_ECHO STDOUT
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
  COMMAND_ERROR_IS_FATAL ANY
)
message("Bare.compile: ${AOTRITON_V2_BUILD_DIR}/Bare.compile")
message("Bare.cluster: ${AOTRITON_V2_BUILD_DIR}/Bare.cluster")
file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.compile" HSACO_RULES)
set(ALL_HSACOS "")

## Use Bare.compile file to generate custom rules
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
  add_custom_command(OUTPUT "${HSACO}"
    COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
    TRITON_CACHE_DIR=${CMAKE_CURRENT_BINARY_DIR}/triton-cache
    "${VENV_DIR}/bin/python"
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
    "--timeout" "${AOTRITON_GPU_BUILD_TIMEOUT}"
    DEPENDS aotriton_venv_triton
  )
  list(APPEND ALL_HSACOS "${HSACO}")
endforeach(RULE)
## Add a virtual target
add_custom_target(aotriton_v2_compile ALL DEPENDS ${ALL_HSACOS})


# Kernel Storage V2
file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.cluster" CLUSTER_RULES)
set(ALL_AKS2 "")
foreach(RULE IN LISTS CLUSTER_RULES)
  list(POP_FRONT RULE DIR_ARCH)
  list(POP_FRONT RULE DIR_FAMILY)
  list(POP_FRONT RULE DIR_KERNEL)
  list(POP_FRONT RULE FONLY)
  string(JOIN "/" AKS2
    ${AOTRITON_KERNEL_STORAGE_V2_DIR}
    ${DIR_ARCH}
    ${DIR_FAMILY}
    ${DIR_KERNEL}
    "${FONLY}.aks2")
  message(STATUS "AKS2 ${AKS2}")
  add_custom_command(OUTPUT "${AKS2}"
    COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
    "${VENV_DIR}/bin/python"
    -m v2python.aks2
    -o "${AKS2}"
    --
    ${RULE}
    DEPENDS aotriton_v2_compile
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
    COMMAND_EXPAND_LISTS)
  list(APPEND ALL_AKS2 "${AKS2}")
endforeach(RULE)
add_custom_target(aotriton_kernel_storage_v2 ALL DEPENDS ${ALL_AKS2})


set(AOTRITON_SHIM_FLAGS "")
if(AOTRITON_NO_SHARED)
    list(APPEND AOTRITON_SHIM_FLAGS "--archive_only" "${AOTRITON_SHIM_FLAGS}")
endif(AOTRITON_NO_SHARED)
if(AOTRITON_NAME_SUFFIX)
  list(APPEND AOTRITON_SHIM_FLAGS "--library_suffix" "${AOTRITON_NAME_SUFFIX}")
endif()
message(STATUS "AOTRITON_SHIM_FLAGS ${AOTRITON_SHIM_FLAGS}")

execute_process(
  COMMAND ${CMAKE_COMMAND}
  -E env VIRTUAL_ENV=${VENV_DIR}
  AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
  "${VENV_DIR}/bin/python" -m v2python.generate_shim --target_gpus ${TARGET_GPUS} --build_dir ${AOTRITON_V2_BUILD_DIR} --bare_mode ${GENERATE_OPTION}
  COMMAND_ECHO STDOUT
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  COMMAND_ERROR_IS_FATAL ANY
)
file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.shim" SHIM_CC_FILES)

# CAVEAT: Actual shim code can only be generated during build phase because it
# requires some kernel information. (Notably shared memory requirement)
add_custom_target(aotriton_v2_gen_shim 
  COMMAND ${CMAKE_COMMAND}
  -E env VIRTUAL_ENV=${VENV_DIR}
  AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
  "${VENV_DIR}/bin/python" -m v2python.generate_shim --target_gpus ${TARGET_GPUS} --build_dir ${AOTRITON_V2_BUILD_DIR} ${AOTRITON_SHIM_FLAGS} ${GENERATE_OPTION}
  BYPRODUCTS ${SHIM_CC_FILES}  # Essential, otherwise add_library complains
  COMMAND_EXPAND_LISTS
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
)
# Note shim and kernel storage v2 are parallel targets
add_dependencies(aotriton_v2_gen_shim aotriton_v2_compile)

if(AOTRITON_NO_SHARED)
  add_library(aotriton_v2 STATIC ${SHIM_CC_FILES})
  set(AOTRITON_LIBRARY_FILE libaotriton${AOTRITON_NAME_SUFFIX}_v2.a)
else(AOTRITON_NO_SHARED)
  add_library(aotriton_v2 SHARED ${SHIM_CC_FILES})
  set(AOTRITON_LIBRARY_FILE libaotriton${AOTRITON_NAME_SUFFIX}_v2.so)
endif(AOTRITON_NO_SHARED)
add_dependencies(aotriton_v2 aotriton_v2_gen_shim)
if(AOTRITON_ENABLE_SUFFIX)
  set_target_properties(aotriton_v2 PROPERTIES OUTPUT_NAME "aotriton${AOTRITON_NAME_SUFFIX}_v2")
endif()

target_include_directories(aotriton_v2 PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../include)
target_include_directories(aotriton_v2 PUBLIC ${CMAKE_BINARY_DIR}/include)  # for <buid>/include/aotriton/config.h. Code should use <aotriton/config.h>
target_include_directories(aotriton_v2 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(aotriton_v2 PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../third_party/incbin)
target_compile_options(aotriton_v2 PRIVATE -fPIC --no-offload-arch=all)
if(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
else(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
endif(AOTRITON_BUILD_FOR_TUNING)
target_link_libraries(aotriton_v2 lzma_interface)

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
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}/include/aotriton"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h")
# A few considerations
# 1. .so file is used for sanity check to ensure all symbols are resolved,
#    but it will not be installed because .a is what you should use to avoid setting
#    search paths for shared libraries
# 2. The archive library will be installed into the hardcode lib/ directory, to avoid Debian vs RHEL divergence.
install(FILES "${AOTRITON_V2_BUILD_DIR}/${AOTRITON_LIBRARY_FILE}" DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/aotriton.images"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

# Python binding only available for AOTriton V2 API
add_library(aotriton INTERFACE)
# add_dependencies(aotriton aotriton_v2)
# target_link_libraries(aotriton INTERFACE ${CMAKE_INSTALL_PREFIX}/lib/libaotriton_v2.a)
# target_include_directories(aotriton INTERFACE ${CMAKE_INSTALL_PREFIX}/include)
# target_link_libraries(aotriton INTERFACE ${AOTRITON_V2_BUILD_DIR}/${AOTRITON_LIBRARY_FILE})
# target_include_directories(aotriton INTERFACE ${CMAKE_CURRENT_SOURCE_PARENT_DIR}/include)
# target_include_directories(aotriton INTERFACE ${CMAKE_CURRENT_SOURCE_PARENT_DIR}/include)
target_link_libraries(aotriton INTERFACE aotriton_v2)
