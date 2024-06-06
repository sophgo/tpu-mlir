
# Ensure necessary environment variables are set
if(NOT DEFINED ENV{CROSS_TOOLCHAINS})
  message(FATAL_ERROR "CROSS_TOOLCHAINS environment variable is not set")
endif()

# Set the C and C++ compilers for ARM
set(CMAKE_C_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER $ENV{CROSS_TOOLCHAINS}/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++)

# Add include directories and set build type
include_directories($ENV{PROJECT_ROOT}/third_party/nntoolchain/include)
include_directories($ENV{PROJECT_ROOT}/third_party/customlayer/include/kernel)
link_directories($ENV{LIBSOPHON_ROOT}/tpu-runtime/build_thirdparty/lib)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
set(CMAKE_BUILD_TYPE Debug)

# Add the atomic_exec library
add_library(atomic_exec SHARED ./atomic_exec.cpp)

# Link the bmlib library
target_link_libraries(atomic_exec PRIVATE bmlib)