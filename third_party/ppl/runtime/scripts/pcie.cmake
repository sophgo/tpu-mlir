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

if(${CHIP} STREQUAL "bm1690" OR ${CHIP} STREQUAL "bm1688")
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc)
else()
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
endif()

if(DEFINED RUNTIME_PATH)
  set(RUNTIME_TOP ${RUNTIME_PATH})
  message(NOTICE "RUNTIME PATH: ${RUNTIME_PATH}")
else()
  if(${CHIP} STREQUAL "bm1690")
    set(RUNTIME_TOP ${PPL_TOP}/runtime/tpuv7-runtime)
  elseif(${CHIP} STREQUAL "bm1684x")
		set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1684x/libsophon/bmlib)
  elseif(${CHIP} STREQUAL "bm1688")
    set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1688/libsophon/bmlib)
  else()
  message(FATAL_ERROR "Unknown chip type:${CHIP}")
  endif()
endif()

# deal extra flags
parse_list("${EXTRA_IDIRS}" EXTRA_IDIRS ";")
parse_list("${EXTRA_LDIRS}" EXTRA_LDIRS ";")
parse_list("${EXTRA_LDFLAGS}" EXTRA_LDFLAGS ";")
parse_list("${EXTRA_CFLAGS}" EXTRA_CFLAGS " ")

set(TPUKERNEL_TOP ${PPL_TOP}/runtime/${CHIP}/TPU1686)
set(KERNEL_TOP ${PPL_TOP}/runtime/kernel)
set(CUS_TOP ${PPL_TOP}/runtime/customize)
include_directories(${TPUKERNEL_TOP}/tpuDNN/include)

include_directories(${TPUKERNEL_TOP}/kernel/include)
include_directories(${KERNEL_TOP})
include_directories(${CUS_TOP}/include)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${RUNTIME_TOP}/include)
include_directories(include)


# Set the library directories for the shared library (link lib${CHIP}.a)
link_directories(${PPL_TOP}/runtime/${CHIP}/lib)
link_directories(${RUNTIME_TOP}/lib)
# Set the output file for the shared library
aux_source_directory(device DEVICE_SRCS)

if(${CHIP} STREQUAL "bm1690")
  set(SHARED_LIBRARY_OUTPUT_FILE libkernel)
  link_directories(${RUNTIME_TOP}/lib)
else()
  set(SHARED_LIBRARY_OUTPUT_FILE lib${CHIP}_kernel_module)
endif()

add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS} ${CUS_TOP}/src/ppl_helper.c)

target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} -Wl,--whole-archive lib${CHIP}.a -Wl,--no-whole-archive m)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  MESSAGE (STATUS "Current is Debug mode")
  SET (FW_DEBUG_FLAGS "-DUSING_FW_DEBUG")
ENDIF ()

# Set the output file properties for the shared library
set_target_properties(${SHARED_LIBRARY_OUTPUT_FILE} PROPERTIES PREFIX "" SUFFIX ".so" COMPILE_FLAGS "-fPIC ${FW_DEBUG_FLAGS}" LINK_FLAGS "-shared")
if(${CHIP} STREQUAL "bm1684x" OR ${CHIP} STREQUAL "bm1688")
  # Set the path to the input file
  set(INPUT_FILE "${CMAKE_BINARY_DIR}/lib${CHIP}_kernel_module.so")
  # Set the path to the output file
  set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/kernel_module_data.h")
  add_custom_command(
    OUTPUT ${KERNEL_HEADER}
    DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE}
    COMMAND echo "const unsigned int kernel_module_data[] = {" > ${KERNEL_HEADER}
    COMMAND hexdump -v -e '1/4 \"0x%08x,\\n\"' ${INPUT_FILE} >> ${KERNEL_HEADER}
    COMMAND echo "}\;" >> ${KERNEL_HEADER})

  # Add a custom target that depends on the custom command
  add_custom_target(gen_kernel_module_data_target DEPENDS ${KERNEL_HEADER})
else()
  install(TARGETS ${SHARED_LIBRARY_OUTPUT_FILE} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
endif()

if(${CHIP} STREQUAL "bm1684x" OR ${CHIP} STREQUAL "bm1688")
  aux_source_directory(host HOST_SRC_FILES)
  # Add a custom target for the shared library
  add_custom_target(dynamic_library DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE})
endif()

message(NOTICE "EXTRA_LDFLAGS: ${EXTRA_LDFLAGS}")
message(NOTICE "EXTRA_IDIRS: ${EXTRA_IDIRS}")
message(NOTICE "EXTRA_CFLAGS: ${EXTRA_CFLAGS}")
message(NOTICE "EXTRA_LDIRS: ${EXTRA_LDIRS}")
find_package(ZLIB REQUIRED)
aux_source_directory(src APP_SRC_FILES)

set(AUTOTUNE_TEST_SRC "")
foreach(file ${APP_SRC_FILES})
    get_filename_component(FILENAME ${file} NAME)
    if(FILENAME STREQUAL "autotune_test.cpp")
        set(AUTOTUNE_TEST_SRC ${file})
        break()
    endif()
endforeach()
if(AUTOTUNE_TEST_SRC)
    message(STATUS "Building autotune_test from ${AUTOTUNE_TEST_SRC}")
    list(REMOVE_ITEM APP_SRC_FILES ${AUTOTUNE_TEST_SRC})
    add_executable(autotest ${AUTOTUNE_TEST_SRC})
    set_target_properties(autotest PROPERTIES OUTPUT_NAME autotune_test)
    install(TARGETS autotest DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(${CHIP} STREQUAL "bm1690")
  aux_source_directory(host PPL_SRC_FILES)
  add_executable(main ${APP_SRC_FILES}
                      ${PPL_SRC_FILES}
                      ${CUS_TOP}/src/cnpy.cpp
                      ${CUS_TOP}/src/host_utils.cpp)
  target_link_libraries(main PRIVATE tpuv7_rt cdm_daemon_emulator tpudnn pthread ${ZLIB_LIBRARIES})
elseif(${CHIP} STREQUAL "bm1684x" OR ${CHIP} STREQUAL "bm1688")
  add_executable(main ${APP_SRC_FILES}
                      ${HOST_SRC_FILES}
                      ${CUS_TOP}/src/cnpy.cpp
                      ${CUS_TOP}/src/host_utils.cpp)
  target_link_libraries(main PRIVATE ${RUNTIME_TOP}/lib/libbmlib.so tpudnn pthread ${ZLIB_LIBRARIES})
endif()
target_include_directories(main PRIVATE ${EXTRA_IDIRS})
target_link_directories(main PRIVATE ${EXTRA_LDIRS})
target_link_libraries(main PRIVATE ${EXTRA_LDFLAGS})
target_compile_options(main PRIVATE ${EXTRA_CFLAGS})
if(${CHIP} STREQUAL "bm1684x" OR ${CHIP} STREQUAL "bm1688")
  add_dependencies(main dynamic_library gen_kernel_module_data_target)
endif()
set_target_properties(main PROPERTIES OUTPUT_NAME test_case)
install(TARGETS main DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
