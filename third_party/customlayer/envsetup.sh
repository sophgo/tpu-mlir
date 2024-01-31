function _rebuild_custom() {
  export CUSTOM_LAYER_DEV_MODE=$1
  echo $CUSTOM_LAYER_PATH
  pushd ${CUSTOM_LAYER_PATH}
  rm -rf build
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" ..
  make
  make install
  popd
}

function _run_custom_test() {
  pushd ${CUSTOM_LAYER_PATH}
  rm -rf build
  mkdir build
  cd build
  cmake -DCMAKE_INSTALL_PREFIX="${CMAKE_CURRENT_BINARY_DIR}" ..
  make test -j4
  popd
}

function rebuild_custom_plugin() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom plugin
}

function rebuild_custom_backend() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom backend
}

function rebuild_custom_firmware_cmodel() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom cmodel
}

function rebuild_custom_firmware_pcie() {
  if [ ! -d "cross_toolchains" ]; then
    mkdir cross_toolchains
  fi
  cd cross_toolchains
  if [ ! -e "gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz" ]; then
    wget https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
    tar -xJvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
  fi
  cd ..
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom pcie
}

function rebuild_custom_firmware_soc() {
  if [ ! -d "cross_toolchains" ]; then
    mkdir cross_toolchains
  fi
  cd cross_toolchains
  if [ "${CUSTOM_LAYER_CHIP_ARCH}" = "bm1684x" ]; then
    if [ ! -e "gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz" ]; then
      wget https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
      tar -xJvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
    fi
  fi
  if [ "${CUSTOM_LAYER_CHIP_ARCH}" = "bm1688" ]; then
    if [ ! -e "Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz" ]; then
      echo "download Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz from https://www.xrvm.cn/community/download?id=4224193099938729984 to directory customlayer/cross_toolchains"
      echo "then use command: tar -xzvf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz"
      return
    fi
  fi
  cd ..
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom soc
}

function rebuild_custom_lib_cmodel() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom plugin
  _rebuild_custom backend
  _rebuild_custom cmodel
}

function rebuild_custom_lib_pcie() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom plugin
  _rebuild_custom backend
  _rebuild_custom pcie
}

function rebuild_custom_lib_soc() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom plugin
  _rebuild_custom backend
  _rebuild_custom soc
}

function run_custom_unittest() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  export CUSTOM_LAYER_DEV_MODE=unittest
  _run_custom_test
}

export CROSS_TOOLCHAINS_DIR="${PWD}/cross_toolchains"
export TPUKERNEL_CUSTOM_FIRMWARE_PATH="${TPUC_ROOT}/lib/libcmodel_custom.so"
export CUSTOM_LAYER_UNITTEST_DIR="${PWD}/test_if/unittest"
