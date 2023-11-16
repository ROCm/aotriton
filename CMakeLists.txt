cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Oort CXX C)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(VENV_DIR "${CMAKE_CURRENT_BINARY_DIR}/venv" CACHE STRING "Virtual Environment Directory")
set(OORT_HIPCC_PATH "hipcc" CACHE STRING "Set HIPCC Path")
set(TARGET_ARCH "MI200" CACHE STRING "Target Architecture")
set(LD_PRELOAD "/opt/rocm/lib/libamdocl64.so:$ENV{LD_PRELOAD}" CACHE STRING "Workaround of libamdhip64.so.5: undefined symbol: hsa_amd_memory_async_copy_on_engine")

# Not a target, we need to override Python3_EXECUTABLE later
execute_process(COMMAND "${Python3_EXECUTABLE}" -m venv "${VENV_DIR}")

set(ENV{VIRTUAL_ENV} "${VENV_DIR}")
# set(Python3_FIND_VIRTUALENV FIRST)
# unset(Python3_EXECUTABLE)
# find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python -m site --user-site OUTPUT_VARIABLE VENV_SITE)
message("VENV_SITE ${VENV_SITE}")

add_custom_target(oort_venv_req
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python -m pip install -r "${CMAKE_CURRENT_LIST_DIR}/requirements.txt"
  BYPRODUCTS "${VENV_DIR}/bin/pytest"
)

set(TRITON_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/triton_build")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${TRITON_BUILD_DIR}")
add_custom_target(oort_venv_triton
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} TRITON_BUILD_DIR=${TRITON_BUILD_DIR} python -m pip install -e . 
  # COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python -m pip show triton
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/third_party/triton/python/"
  BYPRODUCTS "${VENV_SITE}/triton/_C/libtriton.so"
  )
add_dependencies(oort_venv_triton oort_venv_req)

message("CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}")
set(AOTRITON_DIR "${CMAKE_CURRENT_BINARY_DIR}/aotriton")
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory "${AOTRITON_DIR}")

add_custom_target(oort_gen_compile
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python python/generate.py --target ${TARGET_ARCH} --build_dir "${AOTRITON_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  BYPRODUCTS "${AOTRITON_DIR}/Makefile.compile"
)
add_dependencies(oort_gen_compile oort_venv_triton)

include(ProcessorCount)
ProcessorCount(NPROC)
add_custom_target(oort_compile 
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} LD_PRELOAD=${LD_PRELOAD} make -j ${NPROC} -f Makefile.compile
  WORKING_DIRECTORY "${AOTRITON_DIR}"
  BYPRODUCTS "${AOTRITON_DIR}/attn_fwd.h"
  "${AOTRITON_DIR}/bwd_kernel_dk_dv.h"
  "${AOTRITON_DIR}/bwd_kernel_dq.h"
  "${AOTRITON_DIR}/bwd_preprocess.h"
)
add_dependencies(oort_compile oort_gen_compile)

add_custom_target(oort_gen_shim
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} python python/generate_shim.py --build_dir "${AOTRITON_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  BYPRODUCTS "${AOTRITON_DIR}/Makefile.shim"
)
add_dependencies(oort_gen_shim oort_compile) # Shim source files need json metadata

add_custom_target(oort
  ALL
  COMMAND ${CMAKE_COMMAND} -E env VIRTUAL_ENV=${VENV_DIR} make -j ${NPROC} -f Makefile.shim HIPCC=${OORT_HIPCC_PATH}
  WORKING_DIRECTORY "${AOTRITON_DIR}"
  BYPRODUCTS "${AOTRITON_DIR}/liboort.so"
)
add_dependencies(oort oort_gen_shim)

include(GNUInstallDirs)
message("CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}")
install(FILES "${AOTRITON_DIR}/attn_fwd.h"
  "${AOTRITON_DIR}/bwd_kernel_dk_dv.h"
  "${AOTRITON_DIR}/bwd_kernel_dq.h"
  "${AOTRITON_DIR}/bwd_preprocess.h"
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/oort)
install(FILES "${AOTRITON_DIR}/liboort.so" TYPE LIB)
