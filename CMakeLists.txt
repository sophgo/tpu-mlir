cmake_minimum_required(VERSION 3.12)

if (POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(nntoolchain LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_program(CCACHE ccache)
if(CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE})
endif()
#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
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
include_directories(${CMAKE_INSTALL_PREFIX}/include)
message(STATUS "Using LLVM_INCLUDE_DIRS in: ${LLVM_INCLUDE_DIRS}")
message(STATUS "Using MLIR_INCLUDE_DIRS in: ${MLIR_INCLUDE_DIRS}")
#-------------------------------------------------------------------------------
# oneDNN Configuration
#-------------------------------------------------------------------------------
find_package(DNNL REQUIRED)
include_directories(${DNNL_INCLUDE_DIRS})
link_libraries(${DNNL_LIBS})
#-------------------------------------------------------------------------------
# cnpy Configuration
#-------------------------------------------------------------------------------
set(CNPY_PATH ${PROJECT_SOURCE_DIR}/third_party/cnpy)
include_directories(${CNPY_PATH})
#-------------------------------------------------------------------------------
# flatbuffers Configuration
#-------------------------------------------------------------------------------
find_package(Flatbuffers REQUIRED)
include_directories(${FLATBUFFERS_INCLUDE_DIRS})
link_libraries(${FLATBUFFERS_LIBS})
#-------------------------------------------------------------------------------
# nntoolchain Configuration
#-------------------------------------------------------------------------------
set(NNTOOLCHAIN_PATH ${PROJECT_SOURCE_DIR}/third_party/nntoolchain)
include_directories(${NNTOOLCHAIN_PATH}/include)
link_directories(${NNTOOLCHAIN_PATH}/lib)
#-------------------------------------------------------------------------------
# CV18xx Configuration
#-------------------------------------------------------------------------------
set(CV18XX_PATH ${PROJECT_SOURCE_DIR}/third_party/CV18xx)
include_directories(${CV18XX_PATH}/include)
link_directories(${CV18XX_PATH}/lib)
#-------------------------------------------------------------------------------
# PROGRESSBAR Configuration
#-------------------------------------------------------------------------------
set(PROGRESSBAR_PATH ${PROJECT_SOURCE_DIR}/third_party/progressbar)
include_directories(${PROGRESSBAR_PATH}/include)

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

#-------------------------------------------------------------------------------
function(append value)
  foreach(variable ${ARGN})
    set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
  endforeach(variable)
endfunction()
if(TPUMLIR_USE_LLD)
  append("-fuse-ld=lld"
    CMAKE_EXE_LINKER_FLAGS CMAKE_MODULE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS)
endif()
#-------------------------------------------------------------------------------

# generate version
execute_process(
  COMMAND git describe --tags --always
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RESULT_VARIABLE GIT_EXEC_RESULT
  OUTPUT_VARIABLE GIT_SHORT_HASH)

string(STRIP ${GIT_SHORT_HASH} GIT_SHORT_HASH)
string(FIND ${GIT_SHORT_HASH} "-" iLength)
string(SUBSTRING ${GIT_SHORT_HASH} 0 ${iLength} MAIN_VERSION)
math(EXPR iLength "${iLength} + 1")
string(SUBSTRING ${GIT_SHORT_HASH} ${iLength} -1 PATCH_VERSION)
set(GIT_SHORT_HASH "${MAIN_VERSION}.${PATCH_VERSION}")

string(TIMESTAMP BUILD_TIME "%Y%m%d")
set(MLIR_VERSION "${GIT_SHORT_HASH}-${BUILD_TIME}" CACHE STRING "tpu-mlir version" FORCE)
message(STATUS "tpu-mlir version: ${MLIR_VERSION}")
add_definitions(-DMLIR_VERSION="${MLIR_VERSION}")
#-------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Werror -Wno-unused-result -Wreturn-type -Wunused-variable")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
add_subdirectory(include)
add_subdirectory(third_party)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(bindings)
add_subdirectory(python)
