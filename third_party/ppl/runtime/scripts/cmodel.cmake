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

if(DEFINED DEV_MODE)
  message(NOTICE "DEV_MODE: ${DEV_MODE}")
else()
  message(FATAL_ERROR "Please set -DDEV_MODE to cmodel/pcie/soc")
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
    set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1690/tpuv7-runtime-emulator_0.1.0)
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
elseif(${CHIP} STREQUAL "bm1684x"
      OR ${CHIP} STREQUAL "bm1688")
  include_directories(${RUNTIME_TOP}/include)
  include_directories(${RUNTIME_TOP}/src)
  if(NOT ${DEV_MODE} STREQUAL "cmodel")
    find_package(libsophon REQUIRED)
    include_directories(${LIBSOPHON_INCLUDE_DIRS})
  endif()
else()
  message(FATAL_ERROR "Unknown chip type:${CHIP}")
endif()

# Add chip arch defination
# add_definitions(-D__$ENV{CHIP_ARCH}__)

set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/kernel_module_data.h")
add_custom_command(
    OUTPUT ${KERNEL_HEADER}
    COMMAND echo "const unsigned int kernel_module_data[] = {0}\;" > ${KERNEL_HEADER}
)


# Add a custom target that depends on the custom command
add_custom_target(gen_kernel_module_data_target DEPENDS ${KERNEL_HEADER})

aux_source_directory(host HOST_SRC_FILES)
add_library(host STATIC ${HOST_SRC_FILES})
install(TARGETS host DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_dependencies(host gen_kernel_module_data_target)

find_package(ZLIB REQUIRED)
aux_source_directory(src APP_SRC_FILES)
foreach(app_src IN LISTS APP_SRC_FILES)
  get_filename_component(app ${app_src} NAME_WE)
  message(STATUS "add executable: " ${app})
  add_executable(${app} ${app_src} ${CUS_TOP}/src/cnpy.cpp)
  if(${CHIP} STREQUAL "bm1690")
    target_link_libraries(${app} PRIVATE host tpuv7_rt cdm_daemon_emulator pthread ${ZLIB_LIBRARIES})
  elseif(${CHIP} STREQUAL "bm1684x"
      OR ${CHIP} STREQUAL "bm1688")
    target_link_libraries(${app} PRIVATE host bmlib pthread ${ZLIB_LIBRARIES})
  endif()
  add_dependencies(${app} gen_kernel_module_data_target)
  set_target_properties(${app} PROPERTIES OUTPUT_NAME test_case)
  install(TARGETS ${app} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

aux_source_directory(device KERNEL_SRC_FILES)
add_library(firmware SHARED ${KERNEL_SRC_FILES} ${CUS_TOP}/src/ppl_helper.c)
target_include_directories(firmware PRIVATE
	include
	${PPL_TOP}/include
	${CUS_TOP}/include
	${KERNEL_TOP}
	${TPUKERNEL_TOP}/common/include
	${TPUKERNEL_TOP}/kernel/include
	${TPUKERNEL_CUSTOMIZE_TOP}/include
)

target_link_libraries(firmware PRIVATE ${BMLIB_CMODEL_PATH} m)
set_target_properties(firmware PROPERTIES OUTPUT_NAME cmodel)
install(TARGETS firmware DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
