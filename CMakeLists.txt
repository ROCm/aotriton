cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(AOTriton CXX C)

add_subdirectory(third_party/pybind11)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_CXX_COMPILER hipcc)

set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv" CACHE STRING "Virtual Environment Directory")
set(AOTRITON_HIPCC_PATH "hipcc" CACHE STRING "Set HIPCC Path")
option(AOTRITON_BUILD_V1 "Build AOTriton API V1" OFF) # Compiler aborted when compiling hsaco files
option(AOTRITON_BUILD_V2 "Build AOTriton API V2" ON)
set(TARGET_GPUS "MI200" CACHE STRING "Target Architecture (Note here uses Trade names)")
set(AMDHSA_LD_PRELOAD "/opt/rocm/lib/libhsa-runtime64.so" CACHE STRING "Workaround of libamdhip64.so.5: undefined symbol: hsa_amd_memory_async_copy_on_engine")

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

if(AOTRITON_BUILD_V1)
    add_subdirectory(csrc)
endif(AOTRITON_BUILD_V1)

if(AOTRITON_BUILD_V2)
    add_subdirectory(v2src)

    add_subdirectory(bindings) # FIXME: compile python binding
endif(AOTRITON_BUILD_V2)
