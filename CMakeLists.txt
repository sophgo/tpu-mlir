cmake_minimum_required(VERSION 3.12)

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(nntoolchain LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

set(MLIR_DIR ${PROJECT_SOURCE_DIR}/third_party/llvm/lib/cmake/mlir)

# Fix Me. try to use cmake module to define customized sophgo function.
# reference https://github.com/llvm/circt/blob/main/cmake/modules/AddCIRCT.cmake
set(MLIR_BINARY_DIR ${CMAKE_INSTALL_PREFIX})

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
#include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
message(STATUS "Using LLVM_INCLUDE_DIRS in: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")
#-------------------------------------------------------------------------------
# oneDNN Configuration
#-------------------------------------------------------------------------------
set(DNNL_PATH ${PROJECT_SOURCE_DIR}/third_party/oneDNN)
set(DNNL_DIR ${DNNL_PATH}/lib/cmake/dnnl)
find_package(DNNL REQUIRED CONFIG)
include_directories(${DNNL_PATH}/include)
link_directories(${DNNL_PATH}/lib)
#-------------------------------------------------------------------------------
# cnpy Configuration
#-------------------------------------------------------------------------------
set(CNPY_PATH ${PROJECT_SOURCE_DIR}/third_party/cnpy)
include_directories(${CNPY_PATH})
#-------------------------------------------------------------------------------
# flatbuffers Configuration
#-------------------------------------------------------------------------------
set(FLATBUFFERS_PATH ${PROJECT_SOURCE_DIR}/third_party/flatbuffers)
include_directories(${FLATBUFFERS_PATH}/include)
link_directories(${FLATBUFFERS_PATH}/lib)

OPTION (USE_OpenMP "Use OpenMP" ON)
IF(USE_OpenMP)
  FIND_PACKAGE(OpenMP)
  IF(OPENMP_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  ENDIF()
ENDIF()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

execute_process(
  COMMAND git describe --tags
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_EXEC_RESULT
  OUTPUT_VARIABLE GIT_SHORT_HASH)

string(STRIP ${GIT_SHORT_HASH} GIT_SHORT_HASH)
string(TIMESTAMP BUILD_TIME "%Y%m%d")
set(MLIR_VERSION "${GIT_SHORT_HASH}:${BUILD_TIME}")
message(STATUS "nntoolchain version: ${MLIR_VERSION}")
add_definitions(-DMLIR_VERSION="${MLIR_VERSION}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-maybe-uninitialized -Wno-unused-result")
add_subdirectory(include)
add_subdirectory(third_party)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(bindings)
