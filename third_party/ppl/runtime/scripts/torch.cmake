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

if(DEFINED RUNTIME_PATH)
  set(RUNTIME_TOP ${RUNTIME_PATH})
  message(NOTICE "RUNTIME PATH: ${RUNTIME_PATH}")
else()
  if(${CHIP} STREQUAL "bm1690")
    set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1690/tpuv7-runtime-emulator)
		set(BMLIB_CMODEL_PATH ${RUNTIME_TOP}/lib/libtpuv7_emulator.so)
  elseif(${CHIP} STREQUAL "bm1684x")
		set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1684x/libsophon/bmlib)
		set(BMLIB_CMODEL_PATH ${PPL_TOP}/runtime/bm1684x/lib/libcmodel_firmware.so)
  elseif(${CHIP} STREQUAL "bm1688")
    set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1688/libsophon/bmlib)
		set(BMLIB_CMODEL_PATH ${PPL_TOP}/runtime/bm1688/lib/libcmodel_firmware.so)
  else()
  message(FATAL_ERROR "Unknown chip type:${CHIP}")
  endif()
endif()

set(TPUKERNEL_TOP ${PPL_TOP}/runtime/${CHIP}/TPU1686)
set(KERNEL_TOP ${PPL_TOP}/runtime/kernel)
set(CUS_TOP ${PPL_TOP}/runtime/customize)

include_directories(include)
include_directories(${TPUKERNEL_TOP}/kernel/include)
include_directories(${KERNEL_TOP})
include_directories(${CUS_TOP}/include)
include_directories(${CMAKE_BINARY_DIR})
link_directories(${PPL_TOP}/runtime/${CHIP}/lib)
link_directories(${RUNTIME_TOP}/lib)

if(${CHIP} STREQUAL "bm1690")
  include_directories(${RUNTIME_TOP}/include)
elseif(${CHIP} STREQUAL "bm1684x" OR ${CHIP} STREQUAL "bm1688")
  include_directories(${RUNTIME_TOP}/include)
  include_directories(${RUNTIME_TOP}/src)
else()
  message(FATAL_ERROR "Unknown chip type:${CHIP}")
endif()

aux_source_directory(host HOST_SRC_FILES)
aux_source_directory(device KERNEL_SRC_FILES)
add_library(tpu_kernel SHARED ${HOST_SRC_FILES}
															${KERNEL_SRC_FILES}
															${CUS_TOP}/src/ppl_helper.c)
set_target_properties(tpu_kernel PROPERTIES OUTPUT_NAME ${OUT_NAME})
target_include_directories(tpu_kernel PRIVATE
	include
	${PPL_TOP}/include
	${CUS_TOP}/include
	${KERNEL_TOP}
	${TPUKERNEL_TOP}/common/include
	${TPUKERNEL_TOP}/kernel/include
	${TPUKERNEL_CUSTOMIZE_TOP}/include
)

target_link_libraries(tpu_kernel PRIVATE ${BMLIB_CMODEL_PATH} m)
install(TARGETS tpu_kernel DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
