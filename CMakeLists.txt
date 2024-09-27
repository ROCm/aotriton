# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(AOTriton CXX C)
set(AOTRITON_MIN_PYTHON 3.8 CACHE STRING "Minimal Python version for find_package.")
if(AOTRITON_MIN_PYTHON VERSION_LESS "3.8")
  message(FATAL_ERROR "Do not set AOTRITON_MIN_PYTHON lower than 3.8. The code itself does not support it.")
endif()

add_subdirectory(third_party/pybind11)
find_package(Python3 ${AOTRITON_MIN_PYTHON} COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_COMPILER hipcc)

set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv" CACHE STRING "Virtual Environment Directory")
set(AOTRITON_HIPCC_PATH "hipcc" CACHE STRING "Set HIPCC Path")
option(AOTRITON_NO_SHARED "Build as archive library." OFF)
option(AOTRITON_NO_PYTHON "Disable python binding build" OFF)
option(AOTRITON_ENABLE_ASAN "Enable Address Sanitizer. Implies -g" OFF)
option(AOTRITON_BUILD_FOR_TUNING "Build all GPU kernels and set -DAOTRITON_BUILD_FOR_TUNING=1 (=0 otherwise)" OFF)
option(AOTRITON_ENABLE_FP32_INPUTS "Enable FP32 support." ON)
set(AOTRITON_GPU_BUILD_TIMEOUT "8.0" CACHE STRING "GPU kernel compiler times out after X minutes. 0 for indefinite. Highly recommended if AOTRITON_BUILD_FOR_TUNING=On.")
set(TARGET_GPUS "MI200;MI300X;Navi31" CACHE STRING "Target Architecture (Note here uses Trade names)")

# Early failure for python binding
if(NOT AOTRITON_NO_PYTHON)
  find_package(Python3 ${AOTRITON_MIN_PYTHON} COMPONENTS Development REQUIRED)
endif()

# GPU kernel compression related options
option(AOTRITON_COMPRESS_KERNEL "Enable GPU kernel compression with zstd. Fail when zstd is unavailable. Only effective for AOTriton API V2" ON)
# option(AOTRITON_COMPRESS_KERNEL_STATIC_ZSTD "Use static zstd library to avoid potential zstd version conflict (e.g. pytorch)" OFF)

# Resolve name conflicts with suffix
set(AOTRITON_NAME_SUFFIX "" CACHE STRING "Add suffix to namespace and library file name. This is to resolve name conflicts with PyTorch's AOTriton during testing.")
if(AOTRITON_NAME_SUFFIX)
  set(AOTRITON_ENABLE_SUFFIX ON)
else()
  set(AOTRITON_ENABLE_SUFFIX OFF)
endif()
configure_file(include/aotriton/config.h.in include/aotriton/config.h)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/aotriton/config.h
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/aotriton)

# Note for archive library user:
# get this property with:
#   get_property(ZSTD_INCLUDE_DIR TARGET zstd::libzstd_shared PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
# "zstd::libzstd_shared" can be replaced with zstd::libzstd_static
set(AOTRITON_OVERRIDE_ZSTD_INCLUDE "" CACHE STRING "(For archive library users) override zstd header directory.\
Caveat: should consider set AOTRITON_NO_SHARED because objects are compiled with this header file,\
but shared objects will be linked to libzstd found by find_package later.")
if(AOTRITON_COMPRESS_KERNEL)
    find_program(ZSTD_EXEC zstd REQUIRED)
    include(FindPkgConfig)
    pkg_search_module(ZSTD REQUIRED libzstd)
    # if (AOTRITON_COMPRESS_KERNEL_STATIC_ZSTD)
    #     set(ZSTD_TARGET zstd::libzstd_static)
    # else()
    #     if(TARGET zstd::libzstd_shared)
    #         set(ZSTD_TARGET zstd::libzstd_shared)
    #     else()
    #         set(ZSTD_TARGET zstd::libzstd_static)
    #     endif()
    # endif()
    # get_property(AOTRITON_ZSTD_INCLUDE TARGET ${ZSTD_TARGET} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    add_library(zstd_interface INTERFACE)
    # There are other options but these are sufficient on EL8
    target_link_libraries(zstd_interface INTERFACE ${ZSTD_LIBRARIES})
    target_link_directories(zstd_interface INTERFACE ${ZSTD_LIBRARY_DIRS})
    target_include_directories(zstd_interface INTERFACE ${ZSTD_INCLUDE_DIRS})
    set(AOTRITON_ZSTD_INCLUDE "${ZSTD_INCLUDE_DIRS}")
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
message("VENV_DIR ${VENV_DIR}")
# set(Python3_FIND_VIRTUALENV FIRST)
# unset(Python3_EXECUTABLE)
# find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python" -c "import site; print(site.getsitepackages()[0], end='')" OUTPUT_VARIABLE VENV_SITE)
# string(STRIP "${VENV_SITE}" VENV_SITE)
message("VENV_SITE ${VENV_SITE}")

execute_process(COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} "${VENV_DIR}/bin/python" -m pip install -r "${CMAKE_CURRENT_LIST_DIR}/requirements.txt")

set(TRITON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton_build")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TRITON_BUILD_DIR}")
set(AOTRITON_TRITON_SO "${CMAKE_CURRENT_LIST_DIR}/third_party/triton/python/triton/_C/libtriton.so")
set(AOTRITON_TRITON_EGGLINK "${VENV_SITE}/triton.egg-link")
message("AOTRITON_TRITON_EGGLINK ${AOTRITON_TRITON_EGGLINK}")

add_custom_command(OUTPUT "${AOTRITON_TRITON_EGGLINK}"
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} TRITON_BUILD_DIR=${TRITON_BUILD_DIR} "${VENV_DIR}/bin/python" setup.py develop
  # COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python -m pip show triton
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/third_party/triton/python/"
  BYPRODUCTS "${AOTRITON_TRITON_SO}"
)
add_custom_target(aotriton_venv_triton ALL DEPENDS ${AOTRITON_TRITON_EGGLINK})

add_subdirectory(v2src)

if(NOT AOTRITON_NO_PYTHON)
  add_subdirectory(bindings) # FIXME: compile python binding
endif()
