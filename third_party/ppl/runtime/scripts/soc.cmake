cmake_minimum_required(VERSION 3.5)
project(TPUKernelSamples LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
        COMMAND bash -c "FIRMWARE_CHIPID=${CHIP} source $ENV{PPL_PROJECT_ROOT}/samples/scripts/download_toolchain.sh && env"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE output
    )
    if(NOT result EQUAL "0")
        message(FATAL_ERROR "Not able to source download_toolchain.sh: ${output}")
    endif()
    string(REGEX MATCH "CROSS_TOOLCHAINS=([^\n]*)" _ ${output})
    set(ENV{CROSS_TOOLCHAINS} "${CMAKE_MATCH_1}")
    message("CROSS_TOOLCHAINS has been set as $ENV{CROSS_TOOLCHAINS} by default.")
endif()

# Set the C compiler
if(${CHIP} STREQUAL "bm1684x" OR ${CHIP} STREQUAL "bm1690")
  find_program(AARCH64_GCC_FOUND aarch64-linux-gnu-gcc)
  find_program(AARCH64_GPP_FOUND aarch64-linux-gnu-g++)
  if(NOT AARCH64_GCC_FOUND OR NOT AARCH64_GPP_FOUND)
    message(WARNING "aarch64-linux-gnu-gcc or aarch64-linux-gnu-g++ not found.
            Please install the cross-compilation toolchain using:\n
            sudo apt-get install g++-aarch64-linux-gnu gcc-aarch64-linux-gnu")
  endif()
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
  set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
elseif(${CHIP} STREQUAL "bm1688")
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1/bin/riscv64-unknown-linux-gnu-gcc)
  set(CMAKE_ASM_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu/bin/aarch64-none-linux-gnu-g++)
endif()

# Set the include directories for the shared library
set(TPUKERNEL_TOP ${PPL_TOP}/runtime/${CHIP}/TPU1686)
set(KERNEL_TOP ${PPL_TOP}/runtime/kernel)
set(CUS_TOP ${PPL_TOP}/runtime/customize)

if(DEFINED RUNTIME_PATH)
	set(RUNTIME_TOP ${RUNTIME_PATH})
  message(NOTICE "RUNTIME PATH: ${RUNTIME_PATH}")
else()
	if(${CHIP} STREQUAL "bm1690")
		set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1690/tpuv7-runtime-emulator_0.1.0)
	elseif(${CHIP} STREQUAL "bm1684x")
		set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1684x/libsophon/bmlib)
	elseif(${CHIP} STREQUAL "bm1688")
		set(RUNTIME_TOP ${PPL_TOP}/runtime/bm1688/libsophon/bmlib)
	else()
  	message(FATAL_ERROR "Unknown chip type:${CHIP}")
	endif()
endif()

if(NOT DEFINED ENV{SOC_SDK})
  message(FATAL_ERROR "Environment variable SOC_SDK has not been set. Please download libsophon_soc and set SOC_SDK.")
else()
  set(SOC_SDK_INCLUDE_PATH "$ENV{SOC_SDK}/include")
  set(SOC_SDK_LIB_PATH "$ENV{SOC_SDK}/lib")
  if(NOT EXISTS $ENV{SOC_SDK}/include)
    message(FATAL_ERROR "The include path '$ENV{SOC_SDK}/include' does not exist. Please check the path.")
  endif()
  if(NOT EXISTS $ENV{SOC_SDK}/lib)
    message(FATAL_ERROR "The lib path '$ENV{SOC_SDK}/lib' does not exist. Please check the path.")
  endif()
  include_directories($ENV{SOC_SDK}/include)
  link_directories($ENV{SOC_SDK}/lib)
endif()

include_directories(${TPUKERNEL_TOP}/kernel/include)
include_directories(${KERNEL_TOP})
include_directories(${CUS_TOP}/include)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${RUNTIME_TOP}/include)
include_directories(include)

# Set the library directories for the shared library (link lib${CHIP}.a)
link_directories(${PPL_TOP}/runtime/${CHIP}/lib)

set(BM_LIBS bmlib dl pthread)

set(SHARED_LIBRARY_OUTPUT_FILE lib${CHIP}_kernel_module)
aux_source_directory(device DEVICE_SRCS)
set(LIBBMCHIP_PATH "${PPL_TOP}/runtime/${CHIP}/lib/lib${CHIP}.a")

execute_process(
    COMMAND nm -g ${LIBBMCHIP_PATH}
    COMMAND grep "__ppl_get_dtype"
    RESULT_VARIABLE result_dtype
    OUTPUT_VARIABLE output_dtype
)

execute_process(
    COMMAND nm -g ${LIBBMCHIP_PATH}
    COMMAND grep "__dtype"
    RESULT_VARIABLE result_dtype_var
    OUTPUT_VARIABLE output_dtype_var
)

if("${result_dtype}" EQUAL "0" AND "${result_dtype_var}" EQUAL "0")
    message(STATUS "Symbols __ppl_get_dtype and __dtype found in ${LIBBMCHIP_PATH}")
    add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS})
else()
    message(STATUS "One or both symbols __ppl_get_dtype and __dtype not found in ${LIBBMCHIP_PATH}")
    add_library(${SHARED_LIBRARY_OUTPUT_FILE} SHARED ${DEVICE_SRCS} ${CUS_TOP}/src/ppl_helper.c)
endif()

# Link the libraries for the shared library
target_link_libraries(${SHARED_LIBRARY_OUTPUT_FILE} -Wl,--whole-archive lib${CHIP}.a -Wl,--no-whole-archive m)

# Set the output file properties for the shared library
set_target_properties(${SHARED_LIBRARY_OUTPUT_FILE} PROPERTIES PREFIX "" SUFFIX ".so" COMPILE_FLAGS "-fPIC" LINK_FLAGS "-shared")

# Set the path to the input file
set(INPUT_FILE "${CMAKE_BINARY_DIR}/lib${CHIP}_kernel_module.so")

# Set the path to the output file
set(KERNEL_HEADER "${CMAKE_BINARY_DIR}/kernel_module_data.h")
add_custom_command(
    OUTPUT ${KERNEL_HEADER}
    DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE}
    COMMAND echo "const unsigned int kernel_module_data[] = {" > ${KERNEL_HEADER}
    COMMAND hexdump -v -e '1/4 \"0x%08x,\\n\"' ${INPUT_FILE} >> ${KERNEL_HEADER}
    COMMAND echo "}\;" >> ${KERNEL_HEADER}
)

# Add a custom target that depends on the custom command
add_custom_target(gen_kernel_module_data_target DEPENDS ${KERNEL_HEADER})

aux_source_directory(host HOST_SRC_FILES)
add_library(host STATIC ${HOST_SRC_FILES})
install(TARGETS host DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib)
add_dependencies(host gen_kernel_module_data_target)

# Add a custom target for the shared library
add_custom_target(dynamic_library DEPENDS ${SHARED_LIBRARY_OUTPUT_FILE})

# Download and cross compile aarch64 zlib
set(ZLIB_INSTALL_DIR "/ppl/third_party/toolchains_dir/zlib")
file(MAKE_DIRECTORY ${ZLIB_INSTALL_DIR})
set(ZLIB_LIB ${ZLIB_INSTALL_DIR}/lib/libz.a)
if(NOT EXISTS ${ZLIB_LIB})
    add_custom_target(BuildZlib ALL
        COMMAND wget https://zlib.net/zlib-1.3.1.tar.gz -O ${ZLIB_INSTALL_DIR}/zlib.tar.gz
        COMMAND tar -xvf ${ZLIB_INSTALL_DIR}/zlib.tar.gz -C ${ZLIB_INSTALL_DIR}
        COMMAND cd ${ZLIB_INSTALL_DIR}/zlib-1.3.1 && export CROSS_PREFIX=aarch64-linux-gnu- && ./configure --prefix=${ZLIB_INSTALL_DIR} --static
        COMMAND make -C ${ZLIB_INSTALL_DIR}/zlib-1.3.1
        COMMAND make install -C ${ZLIB_INSTALL_DIR}/zlib-1.3.1
        WORKING_DIRECTORY ${ZLIB_INSTALL_DIR}
        COMMENT "Downloading and building zlib"
    )
else()
    message(STATUS "zlib is already installed at ${ZLIB_INSTALL_DIR}.")
endif()

aux_source_directory(src APP_SRC_FILES)
foreach(app_src IN LISTS APP_SRC_FILES)
  get_filename_component(app ${app_src} NAME_WE)
  message(STATUS "add executable: " ${app})
  add_executable(${app} ${app_src} ${CUS_TOP}/src/cnpy.cpp)
  if(${CHIP} STREQUAL "bm1690")
    target_link_libraries(${app} PRIVATE host tpuv7_rt cdm_daemon_emulator pthread ${ZLIB_LIB})
  elseif(${CHIP} STREQUAL "bm1684x"
      OR ${CHIP} STREQUAL "bm1688")
    target_link_libraries(${app} PRIVATE host ${BM_LIBS} pthread  ${ZLIB_LIB})
  endif()
  add_dependencies(${app} dynamic_library gen_kernel_module_data_target)
  set_target_properties(${app} PROPERTIES OUTPUT_NAME test_case)
  install(TARGETS ${app} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
