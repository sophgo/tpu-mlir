cmake_minimum_required(VERSION 2.8.12)

if (POLICY CMP0048)
  # cmake warns if loaded from a min-3.0-required parent dir, so silence the warning:
  cmake_policy(SET CMP0048 NEW)
endif()

project(pybinding)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
SET(CMAKE_INSTALL_RPATH "\${ORIGIN}/")

find_package(pybind11 REQUIRED CONFIG)
pybind11_add_module(pymlir pymlir.cpp)

target_link_libraries(pymlir PRIVATE
  SophgoInitAll
  MLIRTransforms
  MLIRTranslation
  LLVMCore
  LLVMSupport
  LLVMAsmParser
)
install(TARGETS pymlir DESTINATION python)

