# Copyright © 2023-2025 Advanced Micro Devices, Inc.
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

execute_process(COMMAND ${CMAKE_COMMAND}
  -E env VIRTUAL_ENV=${VENV_DIR}
  "${VENV_BIN_PYTHON}" -m v3python.gpu_targets
  --target_arch ${AOTRITON_TARGET_ARCH}
  --target_gpus ${AOTRITON_OVERRIDE_TARGET_GPUS}
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/.."
  OUTPUT_VARIABLE EFFECTIVE_TARGET_GPUS
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "AOTRITON_TARGET_ARCH ${AOTRITON_TARGET_ARCH}")
message(STATUS "AOTRITON_OVERRIDE_TARGET_GPUS ${AOTRITON_OVERRIDE_TARGET_GPUS}")
message(STATUS "EFFECTIVE_TARGET_GPUS ${EFFECTIVE_TARGET_GPUS}")

# Decompress tuning_database.sqlite.xz
execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf
  "${CMAKE_SOURCE_DIR}/v3python/rules/tuning_database.sqlite3.tar.xz"
  WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}")
file(SHA256 "${AOTRITON_V2_BUILD_DIR}/tuning_database.sqlite3" DB_SHA256)

execute_process(COMMAND ${CMAKE_COMMAND} -E tar xf
  "${CMAKE_SOURCE_DIR}/v3python/rules/op_database.sqlite3.tar.xz"
  WORKING_DIRECTORY "${AOTRITON_V2_BUILD_DIR}")
file(SHA256 "${AOTRITON_V2_BUILD_DIR}/op_database.sqlite3" OPDB_SHA256)

get_filename_component(AOTRITON_COMPILER "${CMAKE_CURRENT_LIST_DIR}/../v3python/compile.py" ABSOLUTE)
message("AOTRITON_COMPILER ${AOTRITON_COMPILER}")

if(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "--build_for_tuning")
else(AOTRITON_BUILD_FOR_TUNING)
  set(GENERATE_OPTION "")
endif()

if(AOTRITON_BUILD_FOR_TUNING_BUT_SKIP_KERNEL)
  if (NOT AOTRITON_BUILD_FOR_TUNING)
    message(FATAL_ERROR "AOTRITON_BUILD_FOR_TUNING_BUT_SKIP_KERNEL is only relevant with AOTRITON_BUILD_FOR_TUNING=ON")
  endif()
  list(APPEND GENERATE_OPTION "--build_for_tuning_but_skip_kernel")
  list(APPEND GENERATE_OPTION ${AOTRITON_BUILD_FOR_TUNING_BUT_SKIP_KERNEL})
endif()

if(AOTRITON_NOIMAGE_MODE)
  list(APPEND GENERATE_OPTION "--noimage_mode")
endif()

if(WIN32)
  find_package(dlfcn-win32 REQUIRED)
  set(CMAKE_DL_LIBS dlfcn-win32::dl)
endif()

if(AOTRITON_ENABLE_FP32_INPUTS)
  set(AOTRITON_ENABLE_FP32 1)
else()
  set(AOTRITON_ENABLE_FP32 0)
endif()

## Generate Bare.* files and shim files
execute_process(
COMMAND ${CMAKE_COMMAND}
-E env VIRTUAL_ENV=${VENV_DIR}
AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
"${VENV_BIN_PYTHON}" -X utf8 -m v3python.generate
--target_gpus ${EFFECTIVE_TARGET_GPUS}
--build_dir "${AOTRITON_V2_BUILD_DIR}"
${GENERATE_OPTION}
COMMAND_ECHO STDOUT
WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
COMMAND_ERROR_IS_FATAL ANY
)

# Compile HSACO Kernels
if(NOT AOTRITON_NOIMAGE_MODE)
  set(SIGNATURE_FILE "${AOTRITON_V2_BUILD_DIR}/__signature__.json")
  add_custom_command(OUTPUT "${SIGNATURE_FILE}"
    COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR}
    "${VENV_BIN_PYTHON}"
    -m v3python.write_image_signature
    "${AOTRITON_V2_BUILD_DIR}" ${AOTRITON_GIT_SHA1} "${SIGNATURE_FILE}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
  )

  message("Bare.compile: ${AOTRITON_V2_BUILD_DIR}/Bare.compile")
  message("Bare.cluster: ${AOTRITON_V2_BUILD_DIR}/Bare.cluster")
  set(AOTRITON_HSACO_RECORD "${AOTRITON_V2_BUILD_DIR}/Bare.targets")
  file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.compile" HSACO_RULES ENCODING UTF-8)
  list(LENGTH HSACO_RULES NUM_HSACO_RULES)
  message("Bare.compile ${NUM_HSACO_RULES} rules read")

  ## Use Bare.compile file to generate custom rules
  file(WRITE "${AOTRITON_HSACO_RECORD}" "")
  set(I "0")  # Progress
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
      TRITON_CACHE_DIR=${CMAKE_BINARY_DIR}/triton-cache
      TRITON_F32_DEFAULT="ieee"
      TRITON_STORE_BINARY_ONLY=1
      "${VENV_BIN_PYTHON}"
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
    # DO NOT USE list(APPEND). It is quadratic growth
    file(APPEND "${AOTRITON_HSACO_RECORD}" "${HSACO}" "\n")
    message(CONFIGURE_LOG "hsaco progress ${I}/${NUM_HSACO_RULES}")
  endforeach(RULE)
  ## Add a virtual target
  ### FIXME: AOTRITON_HSACO_RECORD file is duplicated from Bare.compile
  file(STRINGS "${AOTRITON_HSACO_RECORD}" ALL_HSACOS ENCODING UTF-8)
  add_custom_target(aotriton_v2_compile ALL DEPENDS ${ALL_HSACOS})

  # Kernel Storage V2
  set(ALL_AKS2 "")
  set(ALL_ARCH_DIRS "")
  macro(ADD_FROM_CLUSTER_RULES)
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
        "${VENV_BIN_PYTHON}"
        -m v3python.aks2
        -o "${AKS2}"
        ${AKS2_OPTIONS}
        --
        ${RULE}
        DEPENDS aotriton_v2_compile
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
        COMMAND_EXPAND_LISTS)
      list(APPEND ALL_AKS2 "${AKS2}")
      # Add signature file as part of build process to avoid race conditions
      if(NOT "${DIR_ARCH}" IN_LIST ALL_ARCH_DIRS)
        string(JOIN "/" IMAGE_ARCH_SIGNATURE
          ${AOTRITON_KERNEL_STORAGE_V2_DIR}
          ${DIR_ARCH}
          "__signature__")
        add_custom_command(OUTPUT "${IMAGE_ARCH_SIGNATURE}"
          COMMAND ${CMAKE_COMMAND} -E copy "${SIGNATURE_FILE}" "${IMAGE_ARCH_SIGNATURE}"
          DEPENDS "${AKS2}" "${SIGNATURE_FILE}"
          VERBATIM
        )
        list(APPEND ALL_AKS2 "${IMAGE_ARCH_SIGNATURE}")
        list(APPEND ALL_ARCH_DIRS "${DIR_ARCH}")
      endif()
    endforeach(RULE)
  endmacro()
  file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.cluster" CLUSTER_RULES ENCODING UTF-8)
  set(AKS2_OPTIONS "")
  ADD_FROM_CLUSTER_RULES()
  # Affine kernels
  if(EXISTS "${AOTRITON_V2_BUILD_DIR}/Affine.cluster")
    file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Affine.cluster" CLUSTER_RULES ENCODING UTF-8)
    set(AKS2_OPTIONS "--ignore_json")
    ADD_FROM_CLUSTER_RULES()
  endif()
  add_custom_target(aotriton_kernel_storage_v3 ALL DEPENDS ${ALL_AKS2})
endif(NOT AOTRITON_NOIMAGE_MODE)

# The build logic is slightly different when AOTRITON_BUILD_FOR_TUNING=ON
# Some GPU kernels may fail to compile and autotune files should be re-generated accordingly
if(AOTRITON_BUILD_FOR_TUNING)
  # Regenerate with --ignore_missing_kernels
  add_custom_target(aotriton_v2_regen_shim
    COMMAND ${CMAKE_COMMAND}
    -E env VIRTUAL_ENV=${VENV_DIR}
    AOTRITON_ENABLE_FP32=${AOTRITON_ENABLE_FP32}
    "${VENV_BIN_PYTHON}" -m v3python.generate
    --target_gpus ${EFFECTIVE_TARGET_GPUS}
    --build_dir "${AOTRITON_V2_BUILD_DIR}"
    ${GENERATE_OPTION}
    --build_for_tuning_second_pass
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}")
  if(NOT AOTRITON_NOIMAGE_MODE)
    add_dependencies(aotriton_v2_regen_shim aotriton_v2_compile)
  endif(NOT AOTRITON_NOIMAGE_MODE)
endif(AOTRITON_BUILD_FOR_TUNING)

set(AOTRITON_SHIM_FLAGS "")
if(AOTRITON_NAME_SUFFIX)
  list(APPEND AOTRITON_SHIM_FLAGS "--library_suffix" "${AOTRITON_NAME_SUFFIX}")
endif()
message(STATUS "AOTRITON_SHIM_FLAGS ${AOTRITON_SHIM_FLAGS}")

file(STRINGS "${AOTRITON_V2_BUILD_DIR}/Bare.shim" SHIM_CC_FILES ENCODING UTF-8)

aux_source_directory(. CC_FILES)
aux_source_directory(flash/ FLASH_CC_FILES)
add_library(aotriton_v2 SHARED ${SHIM_CC_FILES} ${CC_FILES} ${FLASH_CC_FILES})
set_target_properties(aotriton_v2 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(aotriton_v2 PRIVATE ${CMAKE_DL_LIBS})
if(AOTRITON_ENABLE_SUFFIX)
  set_target_properties(aotriton_v2 PROPERTIES OUTPUT_NAME "aotriton${AOTRITON_NAME_SUFFIX}_v2")
endif()

target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../include>)
# for <build>/include/aotriton/config.h. Code should use <aotriton/config.h>
target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
target_include_directories(aotriton_v2 INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(aotriton_v2 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(aotriton_v2 PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
if(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
  add_dependencies(aotriton_v2 aotriton_v2_regen_shim)
else(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(aotriton_v2 PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
endif(AOTRITON_BUILD_FOR_TUNING)
target_link_libraries(aotriton_v2 PRIVATE lzma_interface)
target_link_libraries(aotriton_v2 PUBLIC hip::host hip::amdhip64)
set_target_properties(aotriton_v2 PROPERTIES VERSION
  ${AOTRITON_VERSION_MAJOR_INT}.${AOTRITON_VERSION_MINOR_INT}.${AOTRITON_VERSION_PATCH_INT})

if(NOT WIN32)
  execute_process(
    COMMAND ${CMAKE_COMMAND}
    -E env VIRTUAL_ENV=${VENV_DIR}
    "${VENV_BIN_PYTHON}" -m v3python.ld_script
    -o "${AOTRITON_V2_BUILD_DIR}/set_aotriton_version.ld"
    ${AOTRITON_VERSION_MAJOR_INT}
    ${AOTRITON_VERSION_MINOR_INT}
    ${AOTRITON_VERSION_PATCH_INT}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_PARENT_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  target_link_options(aotriton_v2 PRIVATE
                      -T "${AOTRITON_V2_BUILD_DIR}/set_aotriton_version.ld")
endif()
  # Otherwise the binary size blows up
  # FIXME: Properly export symbols
  set_target_properties(aotriton_v2 PROPERTIES CXX_VISIBILITY_PRESET hidden)

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
