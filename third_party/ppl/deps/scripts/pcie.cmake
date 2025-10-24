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

message(STATUS "DEV_MODE: " ${DEV_MODE})
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

if(DEBUG)
  set(CMAKE_BUILD_TYPE "Debug")
  add_definitions(-DDEBUG)
else()
  set(CMAKE_BUILD_TYPE "Release")
  if(NOT USING_CUDA)
    add_definitions(-O3)
  endif()
endif()

if(NOT DEFINED ENV{PPL_PROJECT_ROOT})
  message(FATAL_ERROR "Please set environ PPL_PROJECT_ROOT to ppl release path")
else()
  set(PPL_TOP $ENV{PPL_PROJECT_ROOT})
  message(NOTICE "PPL_PATH: ${PPL_TOP}")
endif()

if(NOT DEFINED ENV{CROSS_TOOLCHAINS})
  message("CROSS_TOOLCHAINS was not defined, try source download_toolchain.sh")
  execute_process(
    COMMAND bash -c "CHIP=${CHIP} DEV_MODE=${DEV_MODE} source $ENV{PPL_PROJECT_ROOT}/deps/scripts/download_toolchain.sh && env"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
  )
  if(NOT result EQUAL "0")
    message(FATAL_ERROR "Not able to source download_toolchain.sh: ${output}")
  endif()
  string(REGEX MATCH "CROSS_TOOLCHAINS=([^\n]*)" _ ${output})
  set(ENV{CROSS_TOOLCHAINS} "${CMAKE_MATCH_1}")
endif()

if(${CHIP} STREQUAL "tpub_7_1" OR ${CHIP} STREQUAL "tpul_6_0")
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc)
else()
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
endif()

# deal extra flags
parse_list("${EXTRA_IDIRS}" EXTRA_IDIRS ";")
parse_list("${EXTRA_LDIRS}" EXTRA_LDIRS ";")
parse_list("${EXTRA_LDFLAGS}" EXTRA_LDFLAGS ";")
parse_list("${EXTRA_CFLAGS}" EXTRA_CFLAGS " ")

include($ENV{PPL_RUNTIME_PATH}/scripts/GenChipDef.cmake)
include($ENV{PPL_RUNTIME_PATH}/chip/${CHIP}/config_common.cmake)
include_directories(
  include
  ${CMAKE_BINARY_DIR}
  ${CMAKE_BINARY_DIR}/include
  ${KERNEL_TOP}
  ${TPUKERNEL_TOP}/kernel/include
  ${TPUKERNEL_TOP}/tpuDNN/include
  ${CUS_TOP}/dev/utils/include
  ${RUNTIME_TOP}/include
  ${CUS_TOP}/host/include
  ${CUS_TOP}/dev/utils/include
  ${CHECKER}
  ${EXTRA_IDIRS}
)
link_directories(${BACKEND_LIB_PATH} ${RUNTIME_TOP}/lib ${EXTRA_LDIRS})

aux_source_directory(device DEVICE_SRCS)
set(SHARED_LIBRARY_OUTPUT_FILE libkernel)
add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS} ${CUS_TOP}/dev/utils/src/ppl_helper.c)
target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} -Wl,--whole-archive libfirmware_core.a -Wl,--no-whole-archive -Wl,-s dl m)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  set(FW_DEBUG_FLAGS "-DUSING_FW_DEBUG")
  message(STATUS "Current is Debug mode")
endif()

# Set the output file properties for the shared library
set_target_properties(${SHARED_LIBRARY_OUTPUT_FILE} PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  COMPILE_FLAGS "-fPIC ${FW_DEBUG_FLAGS} ${EXTRA_CFLAGS}"
  LINK_FLAGS "-shared"
)
install(TARGETS ${SHARED_LIBRARY_OUTPUT_FILE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(INPUT_FILE "${CMAKE_BINARY_DIR}/${SHARED_LIBRARY_OUTPUT_FILE}.so")
set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/include/kernel_module_data.h")
add_custom_command(
  OUTPUT ${KERNEL_HEADER}
  DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE}
  COMMAND echo "const unsigned int kernel_module_data[] = {" > ${KERNEL_HEADER}
  COMMAND hexdump -v -e '1/4 \"0x%08x,\\n\"' ${INPUT_FILE} >> ${KERNEL_HEADER}
  COMMAND echo "}\;" >> ${KERNEL_HEADER})

# Add a custom target that depends on the custom command
add_custom_target(gen_kernel_module_data_target DEPENDS ${KERNEL_HEADER})
add_custom_target(dynamic_library DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE})

message(NOTICE "EXTRA_LDFLAGS: ${EXTRA_LDFLAGS}")
message(NOTICE "EXTRA_IDIRS: ${EXTRA_IDIRS}")
message(NOTICE "EXTRA_CFLAGS: ${EXTRA_CFLAGS}")
message(NOTICE "EXTRA_LDIRS: ${EXTRA_LDIRS}")
find_package(ZLIB REQUIRED)
aux_source_directory(src APP_SRC_FILES)

# build main test
aux_source_directory(host HOST_SRC_FILES)
add_executable(main ${APP_SRC_FILES}
               ${HOST_SRC_FILES}
               ${CUS_TOP}/host/src/cnpy.cpp
               ${CUS_TOP}/host/src/host_utils.cpp)

add_dependencies(main dynamic_library gen_kernel_module_data_target)
target_link_libraries(main PRIVATE
  ${RUNTIME_LIBS}
  tpudnn
  pthread
  ${ZLIB_LIBRARIES}
  ${EXTRA_LDFLAGS}
)

target_compile_options(main PRIVATE ${EXTRA_CFLAGS})
set_target_properties(main PROPERTIES OUTPUT_NAME test_case)
install(TARGETS main DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
