
# Ensure necessary environment variables are set
if(NOT DEFINED ENV{CROSS_TOOLCHAINS})
  message(FATAL_ERROR "CROSS_TOOLCHAINS environment variable is not set")
endif()

# Set the C and C++ compilers for ARM

set(CROSS_TOOLCHAINS $ENV{CROSS_TOOLCHAINS})
if($ENV{USE_CROSS_TOOLCHAINS})
  set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
  set(CMAKE_CXX_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)
  message(STATUS "--------------->${CMAKE_C_COMPILER}")
endif()

# Add include directories and set build type
include_directories($ENV{PROJECT_ROOT}/third_party/nntoolchain/include)
include_directories($ENV{PROJECT_ROOT}/third_party/customlayer/include/kernel)
link_directories($ENV{LIBSOPHON_ROOT}/tpu-runtime/build_thirdparty/lib)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_BUILD_TYPE Debug)

add_library(atomic_exec_aarch64 SHARED ./atomic_exec.cpp)
target_link_libraries(atomic_exec_aarch64 PRIVATE bmlib)

add_library(atomic_exec_bm1688_aarch64 SHARED ./atomic_exec_bm1688.cpp)
target_link_libraries(atomic_exec_bm1688_aarch64 PRIVATE bmlib)


add_executable(
  atomic_exec_test_bm1688
  atomic_exec_bm1688.cpp
)
target_link_libraries(atomic_exec_test_bm1688 bmlib pthread dl)
