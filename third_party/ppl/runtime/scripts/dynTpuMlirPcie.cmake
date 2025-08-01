cmake_minimum_required(VERSION 3.5)
project(TPUKernelSamples LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

function(parse_list INPUT OUTPUT CHAR)
  string(REGEX REPLACE ":" "${CHAR}" TMP_LIST "${INPUT}")
  set(${OUTPUT} ${TMP_LIST} PARENT_SCOPE)
endfunction()

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wl,--no-undefined")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--no-undefined")

if(NOT DEFINED ENV{PPL_PROJECT_ROOT})
  message(FATAL_ERROR "Please set environ PPL_PROJECT_ROOT to ppl release path")
else()
  set(PPL_TOP $ENV{PPL_PROJECT_ROOT})
  message(NOTICE "PPL_PATH: ${PPL_TOP}")
endif()

if(DEFINED DEV_MODE)
  message(NOTICE "DEV_MODE: ${DEV_MODE}")
else()
  message(FATAL_ERROR "Please set -DDEV_MODE to cmodel/pcie/soc")
endif()

if(NOT DEFINED CHIP)
  message(FATAL_ERROR "Please set -DCHIP to chip type")
else()
  message(NOTICE "CHIP: ${CHIP}")
endif()
# Add chip arch defination
add_definitions(-D__${CHIP}__)

if(${DEV_MODE} STREQUAL "cmodel")
  add_definitions(-DUSING_CMODEL)
  set(NAME_SUFFIX _${CHIP}_cmodel)
else()  # soc pcie
  set(NAME_SUFFIX ${CHIP})
  # try download cross toolchain
  if(NOT DEFINED ENV{CROSS_TOOLCHAINS})
      message("CROSS_TOOLCHAINS was not defined, try source download_toolchain.sh")
      execute_process(
          COMMAND bash -c "CHIP=${CHIP} DEV_MODE=${DEV_MODE} source $ENV{PPL_PROJECT_ROOT}/samples/scripts/download_toolchain.sh && env"
          RESULT_VARIABLE result
          OUTPUT_VARIABLE output
      )
      if(NOT result EQUAL "0")
          message(FATAL_ERROR "Not able to source download_toolchain.sh: ${output}")
      endif()
      string(REGEX MATCH "CROSS_TOOLCHAINS=([^\n]*)" _ ${output})
      set(ENV{CROSS_TOOLCHAINS} "${CMAKE_MATCH_1}")
  endif()

  if(${CHIP} STREQUAL "bm1688")
    set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc)
  else()
    set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
  endif()
  if(NOT EXISTS ${CMAKE_C_COMPILER})
    message(FATAL_ERROR "Compiler not found: ${CMAKE_C_COMPILER}")
  endif()
endif()

if(DEBUG)
  MESSAGE (STATUS "Current is Debug mode")
  set(CMAKE_BUILD_TYPE "Debug")
  set(FW_DEBUG_FLAGS "-DUSING_FW_DEBUG")
  add_definitions(-DDEBUG)
endif ()
else()
  set(CMAKE_BUILD_TYPE "Release")
  add_definitions(-O3)
endif()

# if(DEFINED RUNTIME_PATH)
#   set(RUNTIME_TOP ${RUNTIME_PATH})
#   message(NOTICE "RUNTIME PATH: ${RUNTIME_PATH}")
# else()
#   if(${CHIP} STREQUAL "bm1690")
#     set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1690/tpuv7-runtime-emulator)
# 		set(BMLIB_CMODEL_PATH ${RUNTIME_TOP}/lib/libtpuv7_emulator.so)
#   elseif(${CHIP} STREQUAL "bm1684x")
# 		set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1684x/libsophon/bmlib)
# 		set(BMLIB_CMODEL_PATH ${PPL_TOP}/runtime/bm1684x/lib/libcmodel_firmware.so)
#   elseif(${CHIP} STREQUAL "bm1688")
#     set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1688/libsophon/bmlib)
# 		set(BMLIB_CMODEL_PATH ${PPL_TOP}/runtime/bm1688/lib/libcmodel_firmware.so)
#   else()
#   message(FATAL_ERROR "Unknown chip type:${CHIP}")
#   endif()
# endif()

if(${CHIP} STREQUAL "bm1684x")
  if(${DEV_MODE} STREQUAL "cmodel")
    set(KERNEL_LIB libcmodel_1684x.so)
  else()
    set(KERNEL_LIB libbm1684x_kernel_module.so)
  endif()
elseif(${CHIP} STREQUAL "bm1688")
  if(${DEV_MODE} STREQUAL "cmodel")
    set(KERNEL_LIB libcmodel_1688.so)
  else()
    set(KERNEL_LIB libbmtpulv60_kernel_module.so)
  endif()
else()
  message(FATAL_ERROR "Unknown chip type:${CHIP}")
endif()

# deal extra flags
parse_list("${EXTRA_IDIRS}" EXTRA_IDIRS ";")
parse_list("${EXTRA_LDIRS}" EXTRA_LDIRS ";")
parse_list("${EXTRA_LDFLAGS}" EXTRA_LDFLAGS ";")
parse_list("${EXTRA_CFLAGS}" EXTRA_CFLAGS " ")

set(TPUKERNEL_TOP ${PPL_TOP}/runtime/${CHIP}/TPU1686)
set(KERNEL_TOP ${PPL_TOP}/runtime/kernel)
set(CUS_TOP ${PPL_TOP}/runtime/customize)

include_directories(include)
include_directories(${TPUKERNEL_TOP}/kernel/include)
include_directories(${KERNEL_TOP})
include_directories(${CUS_TOP}/include)
include_directories(${CMAKE_BINARY_DIR})
# Set the library directories for the shared library (link lib${CHIP}.so)
link_directories(${RUNTIME_TOP}/lib)
# Set the output file for the shared library
aux_source_directory(device DEVICE_SRCS)

set(SHARED_LIBRARY_OUTPUT_FILE lib_kernel_module${NAME_SUFFIX})
add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS} ${CUS_TOP}/src/ppl_helper.c)
target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} PRIVATE ${KERNEL_LIB} m)
# Set the output file properties for the shared library
set_target_properties(${SHARED_LIBRARY_OUTPUT_FILE} PROPERTIES PREFIX "" SUFFIX ".so" COMPILE_FLAGS "-fPIC ${FW_DEBUG_FLAGS}" LINK_FLAGS "-shared")
# Set the path to the input file
set(INPUT_FILE "${CMAKE_BINARY_DIR}/lib${CHIP}_kernel_module.so")
# Set the path to the output file
set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/ppl_kernel_module_data${NAME_SUFFIX}.h")
add_custom_command(
  TARGET  ${SHARED_LIBRARY_OUTPUT_FILE}
  POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "const unsigned int kernel_module_data${NAME_SUFFIX}[] = {" > "${KERNEL_HEADER}"
    COMMAND hexdump -v -e "1/4 \"0x%08x,\\n\"" $<TARGET_FILE:${SHARED_LIBRARY_OUTPUT_FILE}> >> "${KERNEL_HEADER}"
    COMMAND ${CMAKE_COMMAND} -E echo "};" >> "${KERNEL_HEADER}"
  COMMENT "Dumping $<TARGET_FILE:${SHARED_LIBRARY_OUTPUT_FILE}> to ${KERNEL_HEADER}"
  VERBATIM
)


install(
  FILES ${KERNEL_HEADER}
  DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include
  PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
)
