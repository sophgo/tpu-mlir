#!/bin/bash
set -e

DEBUG_FLAG=""
OPTIMIZE_FLAG="-DCMAKE_CXX_FLAGS=-O2"

# Check if clang and clang++ are available
set +e
CLANG_PATH=$(command -v clang)  # Path to the C compiler
CLANGXX_PATH=$(command -v clang++)  # Path to the C++ compiler
set -e

if [ -z "$CLANG_PATH" ] || [ -z "$CLANGXX_PATH" ]; then
    # Use gcc and g++ if CLANG is not found
    COMPILER_FLAG="-DCMAKE_C_COMPILER=$(command -v gcc) -DCMAKE_CXX_COMPILER=$(command -v g++)"
else
    COMPILER_FLAG="-DCMAKE_C_COMPILER=$CLANG_PATH -DCMAKE_CXX_COMPILER=$CLANGXX_PATH"
fi

# Parse arguments
for var in "$@"
do
    if [ "$var" = "DEBUG" ]; then
        DEBUG_FLAG="-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-ggdb"
        OPTIMIZE_FLAG=""  # Don't use O2 optimization for debug builds
    fi
    if [ "$var" = "GCC" ]; then
        COMPILER_FLAG="-DCMAKE_C_COMPILER=$(command -v gcc) -DCMAKE_CXX_COMPILER=$(command -v g++)"
    fi
done

BUILD_FLAG="$COMPILER_FLAG $DEBUG_FLAG $OPTIMIZE_FLAG"

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi

echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"
echo "BUILD_FLAG: $BUILD_FLAG"

# prepare install/build dir
mkdir -p $BUILD_PATH

pushd $BUILD_PATH
cmake -G Ninja \
  $BUILD_FLAG \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
  ${PROJECT_ROOT}
cpu_num=`cat /proc/stat | grep cpu[0-9] -c`
cmake --build . --target install -j${cpu_num}
popd

# Clean up some files for release build
if [ "$1" = "RELEASE" ]; then
  # build doc
  ./release_doc.sh
  # strip mlir tools
  pushd $INSTALL_PATH
  find ./ -name "*.so"  ! -name "*_kernel_module.so" | xargs strip
  ls bin/* | xargs strip
  find ./ -name "*.a" | xargs rm
  popd
fi
