cmake_minimum_required(VERSION 3.5)
project(PplJitHost LANGUAGES C CXX)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined")

if(NOT DEFINED ENV{PPL_PROJECT_ROOT})
  message(FATAL_ERROR "Please set environ PPL_PROJECT_ROOT to ppl release path")
else()
  set(PPL_TOP $ENV{PPL_PROJECT_ROOT})
  message(NOTICE "PPL_PATH: ${PPL_TOP}")
endif()

if(DEBUG)
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  set(CMAKE_BUILD_TYPE "Release")
  if(NOT USING_CUDA)
    add_definitions(-O3)
  endif()
endif()


set(TPUKERNEL_TOP ${PPL_TOP}/runtime/${CHIP}/TPU1686)
set(KERNEL_TOP ${PPL_TOP}/runtime/kernel)
set(CUS_TOP ${PPL_TOP}/runtime/customize)

include_directories(../include)
include_directories(${TPUKERNEL_TOP}/kernel/include)
include_directories(${KERNEL_TOP})
include_directories(${CUS_TOP}/include)

# Add chip arch defination
add_definitions(-D__${CHIP}__)

aux_source_directory(host SRC_FILES)
aux_source_directory(src MAIN)
add_executable(test_case ${SRC_FILES} ${MAIN})

install(TARGETS test_case DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
