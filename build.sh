#!/bin/bash
set -e

if [[ -z "$INSTALL_PATH" ]]; then
  echo "Please source envsetup.sh firstly."
  exit 1
fi

echo "BUILD_PATH: $BUILD_PATH"
echo "INSTALL_PATH: $INSTALL_PATH"
echo "BUILD_FLAG: $BUILD_FLAG"

function config_debug()
{
    cmake -G Ninja \
          -B ${BUILD_PATH} \
          -DCMAKE_C_COMPILER=clang \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_CXX_FLAGS=-ggdb \
          -DTPUMLIR_USE_LLD=ON \
          -DTPUMLIR_INCLUDE_TESTS=ON \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
          ${PROJECT_ROOT}
}

function config_release()
{
    cmake -G Ninja \
          -B ${BUILD_PATH} \
          -DCMAKE_C_COMPILER=clang \
          -DCMAKE_BUILD_TYPE="" \
          -DCMAKE_CXX_COMPILER=clang++ \
          -DCMAKE_CXX_FLAGS=-O2 \
          -DTPUMLIR_USE_LLD=ON \
          -DTPUMLIR_INCLUDE_TESTS=ON \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
          ${PROJECT_ROOT}
}


# prepare install/build dir
if [ "$1" = "DEBUG" ]; then
    rm -rf ${INSTALL_PATH}
    config_debug
else
    rm -rf ${INSTALL_PATH}
    config_release
fi

cpu_num=`cat /proc/stat | grep cpu[0-9] -c`
cmake --build $BUILD_PATH --target install -j${cpu_num}

# Clean up some files for release build
if [ "$1" != "DEBUG" ]; then
  # build doc
  ./release_doc.sh > doc.log 2>&1
  if grep -i 'error' doc.log; then
      exit 1
  fi

  # strip mlir tools
  pushd $INSTALL_PATH
  find ./ -name "*.so"  ! -name "*_kernel_module.so" ! -name "*_atomic_kernel.so" | xargs strip
  ls bin/* | xargs strip
  find ./ -name "*.a" | xargs rm
  popd
fi
