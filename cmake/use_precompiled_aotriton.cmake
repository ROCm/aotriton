add_subdirectory(third_party/pybind11)
# Must be set after pybind11, otherwise pybind11 complains
set(CMAKE_CXX_COMPILER hipcc)

include(ExternalProject)
ExternalProject_Add(aotriton_tar
  URL ${AOTRITON_PYTHON_BINDING_FROM_PREBUILT_TARBALL}
  SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
add_library(aotriton INTERFACE)
add_dependencies(aotriton aotriton_tar)
target_link_libraries(aotriton INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball/lib/libaotriton_v2.a)
target_include_directories(aotriton INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/aotriton_tarball/include)

add_subdirectory(bindings) # FIXME: compile python binding
