cmake_minimum_required(VERSION 3.5)
project(TPUKernelSamples LANGUAGES C CXX)


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

if(NOT DEFINED CHIP)
  message(FATAL_ERROR "Please set -DCHIP to chip type")
else()
  message(NOTICE "CHIP: ${CHIP}")
endif()

if(NOT DEFINED OUT_NAME)
  message(FATAL_ERROR "Please set -DOUT_NAME to output so name")
else()
  message(NOTICE "OUT_NAME: ${OUT_NAME}")
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
  ${CMAKE_BINARY_DIR}
  ${TPUKERNEL_TOP}/kernel/include
  ${KERNEL_TOP}
  ${BACKEND_LIB_PATH}
  ${RUNTIME_TOP}/include
  ${CUS_TOP}/host/include
  ${CUS_TOP}/dev/utils/include
  include
)
link_directories(${BACKEND_LIB_PATH} ${RUNTIME_TOP}/lib)

aux_source_directory(device KERNEL_SRC_FILES)
add_library(tpu_kernel SHARED ${KERNEL_SRC_FILES}
															${CUS_TOP}/dev/utils/src/ppl_helper.c)
set_target_properties(tpu_kernel PROPERTIES OUTPUT_NAME ${OUT_NAME})

target_link_libraries(tpu_kernel PRIVATE ${FIRMWARE_CMODEL} m)
install(TARGETS tpu_kernel DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/chip_map.h
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(DEFINED ENV{TORCH_TPU_RUNTIME_PATH})
  message(NOTICE "TORCH_TPU_RUNTIME_PATH: $ENV{TORCH_TPU_RUNTIME_PATH}")
  set(BACKEND_LIB_PATH $ENV{TORCH_TPU_RUNTIME_PATH})
endif()
set_target_properties(tpu_kernel PROPERTIES INSTALL_RPATH "${BACKEND_LIB_PATH}")
