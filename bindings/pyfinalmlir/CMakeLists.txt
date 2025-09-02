cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pyfinalmlir)

SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/")

find_package(nanobind CONFIG REQUIRED)

file(GLOB _sources pyfinalmlir.cpp)

nanobind_add_module(pyfinalmlir NB_STATIC ${_sources})

target_link_libraries(pyfinalmlir PRIVATE
  TPUMLIRInitAll
  MLIRTransforms
  MLIRParser
  LLVMCore
  LLVMSupport
)

install(TARGETS pyfinalmlir DESTINATION python)
