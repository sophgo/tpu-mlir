function _rebuild_custom() {
  export CUSTOM_LAYER_DEV_MODE=$1
  export BUILD_DIR=${2:-"build"}
  echo $CUSTOM_LAYER_PATH
  pushd ${CUSTOM_LAYER_PATH}
  rm -rf $BUILD_DIR
  mkdir $BUILD_DIR
  cd $BUILD_DIR
  cmake -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" ..
  make
  make install
  cd ..
  popd
}

C_COMPILER=/usr/bin/aarch64-linux-gnu-gcc
CXX_COMPILER=/usr/bin/aarch64-linux-gnu-g++
function _rebuild_custom_aarch64() {
  export CUSTOM_LAYER_DEV_MODE=$1
  export BUILD_DIR=${2:-"build"}
  echo $CUSTOM_LAYER_PATH
  pushd ${CUSTOM_LAYER_PATH}
  rm -rf $BUILD_DIR
  mkdir $BUILD_DIR
  cd $BUILD_DIR
  cmake -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" .. -DCMAKE_C_COMPILER=$C_COMPILER  -DCMAKE_CXX_COMPILER=$CXX_COMPILER
  make
  make install
  cd ..
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

function rebuild_custom_cpuop_x86() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom customcpuop build_ap
}
function rebuild_custom_apop_x86() {
  rebuild_custom_cpuop_x86
}

function rebuild_custom_cpuop_aarch64() {
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom_aarch64 customcpuop build_ap
}
function rebuild_custom_apop_aarch64() {
  rebuild_custom_cpuop_aarch64
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
  pushd cross_toolchains
  if [ ! -e "gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz" ]; then
    wget https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
    tar -xJvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
  fi
  popd
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  _rebuild_custom pcie
}

function rebuild_custom_firmware_soc() {
  if [ ! -d "cross_toolchains" ]; then
    mkdir cross_toolchains
  fi
  pushd cross_toolchains
  export CUSTOM_LAYER_CHIP_ARCH=${1:-bm1684x}
  if [ "${CUSTOM_LAYER_CHIP_ARCH}" = "bm1684x" ]; then
    if [ ! -e "gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz" ]; then
      wget https://armkeil.blob.core.windows.net/developer/Files/downloads/gnu-a/10.3-2021.07/binrel/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
      tar -xJvf gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
    fi
  elif [ "${CUSTOM_LAYER_CHIP_ARCH}" = "bm1688" ]; then
    if [ ! -e "Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz" ]; then
      # echo "download Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz from https://www.xrvm.cn/community/download?id=4224193099938729984 to directory customlayer/cross_toolchains"
      # echo "then use command: tar -xzvf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz"
      xuetie_toolchain=https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1663142514282
      toolchain_file_name=Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz
      wget ${xuetie_toolchain}/${toolchain_file_name}
      tar -xzvf Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz
    fi
  fi
  popd
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
