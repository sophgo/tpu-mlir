#!/bin/bash
set -e
function download_toolchain() {
  toolchain=$1
  addr=$2
  filename=$3
  if [ ! -d ${toolchain} ]; then
    if [ ! -e ${filename} ]; then
      wget ${addr}
    fi
    tar xvf $filename
  fi
}

function download_gcc_arm() {
  tool_name="gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
  tool_addr="https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/${tool_name}.tar.xz"
  tool_file=${tool_name}.tar.xz
  download_toolchain $tool_name $tool_addr $tool_file
}

function download_gcc_linaro() {
  tool_name="gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu"
  tool_addr="https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz"
  tool_file=${tool_name}.tar.xz
  download_toolchain $tool_name $tool_addr $tool_file
}

function download_riscv_xuantie900() {
  tool_name="Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1"
  tool_addr="https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1695015316167/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz"
  tool_file="Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz"
  download_toolchain $tool_name $tool_addr $tool_file
}

function download_loong() {
  tool_name="loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1"
  tool_addr="http://ftp.loongnix.cn/toolchain/gcc/release/loongarch/gcc8/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz"
  tool_file="loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz"
  download_toolchain $tool_name $tool_addr $tool_file
}

CROSS_TOOLCHAINS=${1:-$PROJECT_ROOT/../bm_prebuilt_toolchains}
loongarch_host_toolchain=${2}

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

if [ -z "$CROSS_TOOLCHAINS" ]; then
  export CROSS_TOOLCHAINS=$CROSS_TOOLCHAINS
fi
echo "export CROSS_TOOLCHAINS=$CROSS_TOOLCHAINS"

mkdir -p $CROSS_TOOLCHAINS
pushd $CROSS_TOOLCHAINS >>/dev/null

if [ x${CHIP} == "xbm1688" ]; then
  download_riscv_xuantie900
  if [ x${DEV_MODE} == "xsoc" ]; then
    download_gcc_arm
  fi
elif [ x${CHIP} == "xbm1684x" ]; then
  if [ "x${loongarch_host_toolchain}" == "xloongarch64" ]; then
    download_loong
  else
    download_gcc_arm
  fi
  if [ x${DEV_MODE} == "xsoc" ]; then
    download_gcc_linaro
  fi
elif [ x${CHIP} == "xbm1690" ]; then
    download_riscv_xuantie900
else
  return 1
fi
popd >>/dev/null
