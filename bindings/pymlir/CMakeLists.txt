cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pymlir)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/")

if (POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

file(GLOB _sources pymodule.cpp pymlir.cpp host/*.cpp)

if (TPUMLIR_USE_CUDA)
  add_subdirectory(cuda)
  file(GLOB cuda_sources pycuda.cpp cuda/*.cpp)
  list(APPEND _sources ${cuda_sources})
endif ()

nanobind_add_module(pymlir ${_sources})

target_link_libraries(pymlir PRIVATE
  TPUMLIRInitAll
  MLIRTransforms
  MLIRParser
  LLVMCore
  LLVMSupport
)

if (TPUMLIR_USE_CUDA)
  target_link_libraries(pymlir PRIVATE cudnn cudart cuda_helper)
endif()

install(TARGETS pymlir DESTINATION python)
