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

if(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "--build_for_tuning")
else(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "")
endif()

if(AOTRITON_NOIMAGE_MODE)
  list(APPEND GENERATE_OPTION "--noimage_mode")
endif()

if(AOTRITON_ENABLE_FP32_INPUTS)
  set(AOTRITON_ENABLE_FP32 1)
else()
  set(AOTRITON_ENABLE_FP32 0)
endif()

# Compile HSACO Kernels
if(NOT AOTRITON_NOIMAGE_MODE)
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
    # message(STATUS "Add AKS2 ${AKS2}")
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
endif(NOT AOTRITON_NOIMAGE_MODE)

set(AOTRITON_SHIM_FLAGS "")
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

add_custom_target(aotriton_v2_gen_shim 
  COMMAND ${CMAKE_COMMAND}
  -E env VIRTUAL_ENV=${VENV_DIR}
  AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
  "${VENV_DIR}/bin/python" -m v2python.generate_shim --target_gpus ${TARGET_GPUS} --build_dir ${AOTRITON_V2_BUILD_DIR} ${AOTRITON_SHIM_FLAGS} ${GENERATE_OPTION}
  BYPRODUCTS ${SHIM_CC_FILES}  # Essential, otherwise add_library complains
  COMMAND_EXPAND_LISTS
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
)
if(NOT AOTRITON_NOIMAGE_MODE)
  # shim and compile can be parallel targets under Storage v2
  # However we still keep the dependency because we want sanity check to make
  # sure all GPU objects referenced by shim layer have been built
  # (The actual effect of --noimage_mode for generate_shim is to disable this check)
  add_dependencies(aotriton_v2_gen_shim aotriton_v2_compile)
endif()

add_library(aotriton_v2 SHARED ${SHIM_CC_FILES})
set(AOTRITON_LIBRARY_FILE libaotriton${AOTRITON_NAME_SUFFIX}_v2.so)
add_dependencies(aotriton_v2 aotriton_v2_gen_shim)
if(AOTRITON_ENABLE_SUFFIX)
  set_target_properties(aotriton_v2 PROPERTIES OUTPUT_NAME "aotriton${AOTRITON_NAME_SUFFIX}_v2")
endif()

target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>)
# for <build>/include/aotriton/config.h. Code should use <aotriton/config.h>
target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
target_include_directories(aotriton_v2 INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(aotriton_v2 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_options(aotriton_v2 PRIVATE -fPIC)
if(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
else(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
endif(AOTRITON_BUILD_FOR_TUNING)
target_link_libraries(aotriton_v2 PRIVATE lzma_interface)
target_link_libraries(aotriton_v2 PUBLIC hip::host hip::amdhip64)
set_target_properties(aotriton_v2 PROPERTIES VERSION
  ${AOTRITON_VERSION_MAJOR_INT}.${AOTRITON_VERSION_MINOR_INT}.${AOTRITON_VERSION_PATCH_INT})

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}/include/aotriton"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
install(TARGETS aotriton_v2
        EXPORT aotriton-targets
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(EXPORT aotriton-targets
  DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/cmake/aotriton)

# Packages
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/aotriton-config.cmake"
  INSTALL_DESTINATION "lib/cmake/aotriton"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/aotriton-config-version.cmake"
  VERSION "${AOTRITON_VERSION_MAJOR_INT}.${AOTRITON_VERSION_MINOR_INT}.${AOTRITON_VERSION_PATCH_INT}"
  COMPATIBILITY AnyNewerVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/aotriton-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/aotriton-config-version.cmake
  DESTINATION lib/cmake/aotriton)

if(NOT AOTRITON_NOIMAGE_MODE)
  install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/aotriton.images"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif(NOT AOTRITON_NOIMAGE_MODE)

# Python binding only available for AOTriton V2 API
add_library(aotriton INTERFACE)
target_link_libraries(aotriton INTERFACE aotriton_v2)
