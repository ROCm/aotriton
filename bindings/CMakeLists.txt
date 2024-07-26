# Copyright © 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

aux_source_directory(. PYAOTRITON_SRC)
# find_package(hip REQUIRED)
pybind11_add_module(pyaotriton ${PYAOTRITON_SRC})
# target_link_libraries(pyaotriton PRIVATE hip::device)
target_compile_features(pyaotriton PRIVATE cxx_std_20)
message(STATUS "ZSTD_TARGET in bindings/: ${ZSTD_TARGET}")
target_link_libraries(pyaotriton PUBLIC aotriton)
if(AOTRITON_OVERRIDE_ZSTD_LIB)
  target_link_libraries(pyaotriton PRIVATE ${AOTRITON_OVERRIDE_ZSTD_LIB})
else()
  target_link_libraries(pyaotriton PRIVATE ${ZSTD_TARGET})
endif()
# TODO: Unify build option marcos with "interface target+public compile definitions"
if(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(pyaotriton PRIVATE -DAOTRITON_BUILD_FOR_TUNING=1)
else(AOTRITON_BUILD_FOR_TUNING)
  target_compile_definitions(pyaotriton PRIVATE -DAOTRITON_BUILD_FOR_TUNING=0)
endif(AOTRITON_BUILD_FOR_TUNING)
