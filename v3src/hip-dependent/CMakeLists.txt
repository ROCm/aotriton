# Copyright © 2023-2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Every .cc here calls a real hip*() runtime function -- that is what the
# directory means, so the source list is simply whatever is in it.
aux_source_directory(. HIP_CC_FILES)

add_library(aotriton_v2 SHARED ${HIP_CC_FILES})
set_target_properties(aotriton_v2 PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(aotriton_v2 PRIVATE ${CMAKE_DL_LIBS})
if(AOTRITON_ENABLE_SUFFIX)
  set_target_properties(aotriton_v2 PROPERTIES OUTPUT_NAME "aotriton${AOTRITON_NAME_SUFFIX}_v2")
endif()

target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
# for <build>/include/aotriton/config.h. Code should use <aotriton/config.h>
target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>)
target_include_directories(aotriton_v2 INTERFACE $<INSTALL_INTERFACE:include>)
target_include_directories(aotriton_v2 PRIVATE ${AOTRITON_V2_BUILD_DIR})
# for <build>/v3src/include/aotriton/<family>/backends.h -- the per-operator
# backend-index constants aotriton.generate emits. Public because the index is:
# attn_options::force_backend_index takes it.
target_include_directories(aotriton_v2 PUBLIC $<BUILD_INTERFACE:${AOTRITON_V2_BUILD_DIR}/include>)
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
# Static-archive linking drops any .o whose symbols are only referenced from
# outside the library -- i.e. every AOTRITON_API entry point PyTorch calls.
# WHOLE_ARCHIVE forces all objects in, matching the old behavior of compiling
# everything into one SHARED library.
# Needs CMake >= 3.24; root floor is 3.27.
target_link_libraries(aotriton_v2 PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,aotriton_common>)
set_target_properties(aotriton_v2 PROPERTIES VERSION
  ${AOTRITON_VERSION_MAJOR_INT}.${AOTRITON_VERSION_MINOR_INT}.${AOTRITON_VERSION_PATCH_INT})

if(NOT WIN32)
  execute_process(
    COMMAND ${CMAKE_COMMAND}
    -E env VIRTUAL_ENV=${VENV_DIR}
    "${VENV_BIN_PYTHON}" -m aotriton.comment_only_asm
    -o "${AOTRITON_V2_BUILD_DIR}/set_aotriton_version.s"
    ${AOTRITON_VERSION_MAJOR_INT}
    ${AOTRITON_VERSION_MINOR_INT}
    ${AOTRITON_VERSION_PATCH_INT}
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMAND_ERROR_IS_FATAL ANY
  )
  target_sources(aotriton_v2 PRIVATE "${AOTRITON_V2_BUILD_DIR}/set_aotriton_version.s")
else()
  # FIXME: Add version string to Windows binaries.
endif()
# Otherwise the binary size blows up
# FIXME: Properly export symbols
set_target_properties(aotriton_v2 PROPERTIES CXX_VISIBILITY_PRESET hidden)

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/aotriton"
        DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")
# The generated half of the public headers (per-family backends.h), installed
# into the same tree so a consumer sees one include root.
install(DIRECTORY "${AOTRITON_V2_BUILD_DIR}/include/aotriton"
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
  install(DIRECTORY "${AOTRITON_KERNEL_STORAGE_V2_DIR}"
          DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
          PATTERN "*.nsv" EXCLUDE)
endif(NOT AOTRITON_NOIMAGE_MODE)

# Python binding only available for AOTriton V2 API
add_library(aotriton INTERFACE)
target_link_libraries(aotriton INTERFACE aotriton_v2)
