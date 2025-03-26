#!/bin/bash
# ./build.sh [DEBUG|RELEASE] [CUDA|CPU]
set -e

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi

# Function to show usage information
usage() {
  echo "Usage: $0 [RELEASE|DEBUG] [CPU|CUDA]"
}

echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"
#echo "BUILD_FLAG: $BUILD_FLAG"

BUILD_TYPE=""
CXX_FLAGS="-O2"
USE_CUDA="OFF"

if [ -n "$1" ]; then
    if [ "$1" = "DEBUG" ]; then
        BUILD_TYPE="Debug"
        CXX_FLAGS="-ggdb"
    elif [ "$1" != "RELEASE" ]; then
        echo "Invalid build mode: $1"
        usage
        exit 1
    fi
fi

# Check for CUDA support
if [ -n "$2" ]; then
    if [ "$2" = "CUDA" ]; then
        USE_CUDA="ON"
    elif [ "$2" != "CPU" ]; then
        echo "Invalid CUDA option: $2"
        usage
        exit 1
    fi
fi

# prepare install/build dir
rm -rf "${INSTALL_PATH}"
cmake -G Ninja \
  -B "${BUILD_PATH}" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DCMAKE_CXX_FLAGS="${CXX_FLAGS}" \
  -DTPUMLIR_USE_LLD=ON \
  -DTPUMLIR_INCLUDE_TESTS=ON \
  -DTPUMLIR_USE_CUDA="${USE_CUDA}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}" \
  "${PROJECT_ROOT}"

cpu_num=$(cat /proc/stat | grep cpu[0-9] -c)
cmake --build $BUILD_PATH --target install -j${cpu_num}

cmake --build $BUILD_PATH --target passes_json_files builder_python install_passes_files

# build ppl code
rm -rf lib/PplBackend/build
bash lib/PplBackend/build.sh

# Clean up some files for release build
if [ "$1" != "DEBUG" ]; then
  # build doc
  ./release_doc.sh >doc.log 2>&1
  if grep -i 'error' doc.log; then
    exit 1
  fi
  rm doc.log

  # strip mlir tools
  pushd $INSTALL_PATH
  find ./ -name "*.so" ! -name "*_kernel_module.so" ! -name "*_atomic_kernel.so" | xargs strip
  ls bin/* | xargs strip
  find ./ -name "*.a" ! -name "*_kernel_module.a" | xargs rm
  popd
fi
