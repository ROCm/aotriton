# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(AOTriton CXX C)

add_subdirectory(third_party/pybind11)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_COMPILER hipcc)

set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv" CACHE STRING "Virtual Environment Directory")
set(AOTRITON_HIPCC_PATH "hipcc" CACHE STRING "Set HIPCC Path")
option(AOTRITON_BUILD_V2 "Build AOTriton API V2" ON)
option(AOTRITON_NO_SHARED "Disable shared object build. Incompatible with AOTRITON_COMPRESS_KERNEL." ON)
option(AOTRITON_NO_PYTHON "Disable python binding build" OFF)
option(AOTRITON_ENABLE_ASAN "Enable Address Sanitizer. Implies -g" OFF)
set(TARGET_GPUS "MI200;MI300X" CACHE STRING "Target Architecture (Note here uses Trade names)")
set(AMDHSA_LD_PRELOAD "/opt/rocm/lib/libhsa-runtime64.so" CACHE STRING "Workaround of libamdhip64.so.5: undefined symbol: hsa_amd_memory_async_copy_on_engine")

# GPU kernel compression related options
option(AOTRITON_COMPRESS_KERNEL "Enable GPU kernel compression with zstd. Fail when zstd is unavailable. Only effective for AOTriton API V2" ON)
option(AOTRITON_COMPRESS_KERNEL_STATIC_ZSTD "Use static zstd library to avoid potential zstd version conflict (e.g. pytorch)" ON)
# Note for archive library user:
# get this property with:
#   get_property(ZSTD_INCLUDE_DIR TARGET zstd::libzstd_shared PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
# "zstd::libzstd_shared" can be replaced with zstd::libzstd_static
set(AOTRITON_OVERRIDE_ZSTD_INCLUDE "" CACHE STRING "(For archive library users) override zstd header directory.\
Caveat: should consider set AOTRITON_NO_SHARED because objects are compiled with this header file,\
but shared objects will be linked to libzstd found by find_package later.")
set(AOTRITON_OVERRIDE_ZSTD_LIB "" CACHE STRING "(For archive library users) override zstd header library")
if(AOTRITON_COMPRESS_KERNEL)
    find_program(ZSTD_EXEC zstd REQUIRED)
    find_package(zstd REQUIRED)
    if (AOTRITON_COMPRESS_KERNEL_STATIC_ZSTD)
        set(ZSTD_TARGET zstd::libzstd_static)
    else()
        if(TARGET zstd::libzstd_shared)
            set(ZSTD_TARGET zstd::libzstd_shared)
        else()
            set(ZSTD_TARGET zstd::libzstd_static)
        endif()
    endif()
    get_property(AOTRITON_ZSTD_INCLUDE TARGET ${ZSTD_TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "ZSTD_TARGET ${ZSTD_TARGET}")
    message(STATUS "get_property AOTRITON_ZSTD_INCLUDE ${AOTRITON_ZSTD_INCLUDE}")
    if(AOTRITON_OVERRIDE_ZSTD_INCLUDE)
        set(AOTRITON_ZSTD_INCLUDE ${AOTRITON_OVERRIDE_ZSTD_INCLUDE})
    endif()
endif()

set(AOTRITON_EXTRA_COMPILER_OPTIONS "-I/opt/rocm/include/ ")
if(AOTRITON_ENABLE_ASAN)
    set(AOTRITON_EXTRA_COMPILER_OPTIONS "${AOTRITON_EXTRA_COMPILER_OPTIONS} -g -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${AOTRITON_EXTRA_COMPILER_OPTIONS}")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(AOTRITON_EXTRA_COMPILER_OPTIONS "${AOTRITON_EXTRA_COMPILER_OPTIONS} -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(AOTRITON_EXTRA_COMPILER_OPTIONS "${AOTRITON_EXTRA_COMPILER_OPTIONS} ${CMAKE_CXX_FLAGS_RELEASE}")
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(AOTRITON_EXTRA_COMPILER_OPTIONS "${AOTRITON_EXTRA_COMPILER_OPTIONS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

set(Python_ARTIFACTS_INTERACTIVE TRUE)

# Not a target, we need to override Python3_EXECUTABLE later
execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${VENV_DIR}")

set(ENV{VIRTUAL_ENV} "${VENV_DIR}")
# set(Python3_FIND_VIRTUALENV FIRST)
# unset(Python3_EXECUTABLE)
# find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python -m site --user-site OUTPUT_VARIABLE VENV_SITE)
message("VENV_SITE ${VENV_SITE}")

add_custom_target(aotriton_venv_req
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" python -m pip install -r "${CMAKE_CURRENT_LIST_DIR}/requirements.txt"
  BYPRODUCTS "${VENV_DIR}/bin/pytest"
)

set(TRITON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton_build")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TRITON_BUILD_DIR}")
add_custom_target(aotriton_venv_triton
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} PATH="${VENV_DIR}/bin:$ENV{PATH}" TRITON_BUILD_DIR=${TRITON_BUILD_DIR} python setup.py develop
  # COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python -m pip show triton
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/third_party/triton/python/"
  BYPRODUCTS "${VENV_SITE}/triton/_C/libtriton.so"
  )
add_dependencies(aotriton_venv_triton aotriton_venv_req)

if(AOTRITON_BUILD_V2)
    add_subdirectory(v2src)

    if(NOT AOTRITON_NO_PYTHON)
        add_subdirectory(bindings) # FIXME: compile python binding
    endif()
endif(AOTRITON_BUILD_V2)
