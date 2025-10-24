cmake_minimum_required(VERSION 3.5)
project(PplJit LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
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

if(NOT DEFINED CHIP)
  message(FATAL_ERROR "Please set -DCHIP to chip type")
else()
  message(NOTICE "CHIP: ${CHIP}")
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

include($ENV{PPL_RUNTIME_PATH}/scripts/GenChipDef.cmake)
include($ENV{PPL_RUNTIME_PATH}/chip/${CHIP}/config_common.cmake)

include_directories(
  include
  ${CMAKE_BINARY_DIR}
  ${KERNEL_TOP}
  ${TPUKERNEL_TOP}/kernel/include
  ${CUS_TOP}/host/include
  ${CUS_TOP}/dev/utils/include
  ${CHECKER})
link_directories(${BACKEND_LIB_PATH})
# Add chip arch defination
add_definitions(-D__${CHIP}__)

aux_source_directory(device KERNEL_SRC_FILES)
add_library(${NAME} SHARED ${KERNEL_SRC_FILES} ${CUS_TOP}/dev/utils/src/ppl_helper.c ${KERNEL_CHECKER})
target_include_directories(${NAME} PRIVATE
  include
  ${PPL_TOP}/include
  ${CUS_TOP}/include
  ${TPUKERNEL_TOP}/common/include
  ${TPUKERNEL_TOP}/kernel/include
)

if(NOT DEFINED ENV{TPUC_ROOT})
  target_link_libraries(${NAME} PRIVATE ${FIRMWARE_CMODEL} m)
else()
  if(${CHIP} STREQUAL "tpub_7_1" OR ${CHIP} STREQUAL "tpub_9_0")
    set(BMLIB_CMODEL_PATH $ENV{TPUC_ROOT}/lib/libbackend_bm1690.so)
  elseif(${CHIP} STREQUAL "tpu_6_0")
    set(BMLIB_CMODEL_PATH $ENV{TPUC_ROOT}/lib/libbackend_1684x.so)
  elseif(${CHIP} STREQUAL "tpu_6_0_e")
    set(BMLIB_CMODEL_PATH $ENV{TPUC_ROOT}/lib/libbackend_1684xe.so)
  elseif(${CHIP} STREQUAL "tpul_6_0")
    set(BMLIB_CMODEL_PATH $ENV{TPUC_ROOT}/lib/libbackend_1688.so)
  elseif(${CHIP} STREQUAL "tpul_8_1")
    set(BMLIB_CMODEL_PATH $ENV{TPUC_ROOT}/lib/libbackend_mars3.so)
  else()
    message(FATAL_ERROR "Unknown chip type:${CHIP}")
  endif()
  target_link_libraries(${NAME} PRIVATE ${BMLIB_CMODEL_PATH} m)
endif()
install(TARGETS ${NAME} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
