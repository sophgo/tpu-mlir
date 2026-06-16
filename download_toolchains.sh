#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# PPL compiler version and download URL
# Update these when a new PPL release is needed
PPL_URL="https://github.com/sophgo/tpu-mlir/releases/download/v1.28.1/ppl_v1.7.111-g23cf8831-20260616.tar.gz"


function usage() {
  echo "Usage: $0 [ppl|cross-gcc|all] [--dir PATH]"
  echo ""
  echo "Targets:"
  echo "  ppl         Download PPL compiler only"
  echo "  cross-gcc   Download all cross GCC toolchains (arm, linaro, riscv, loong)"
  echo "  all         Download everything (default when no target is specified)"
  echo ""
  echo "Options:"
  echo "  --dir PATH  Set download directory (default: \$DIR/cross_toolchains)"
  echo "  -h, --help  Show this help message"
}

function download_toolchain() {
  local toolchain=$1
  local addr=$2
  local filename=$3
  if [ ! -d "${toolchain}" ]; then
    echo "Downloading ${toolchain}..."
    if [ ! -e "${filename}" ]; then
      wget "${addr}"
    fi
    tar xvf "${filename}"
  else
    echo "${toolchain} already exists, skipping."
  fi
}

function download_ppl() {
  PPL_PACKAGE="${PPL_URL##*/}"
  PPL_VERSION="${PPL_PACKAGE#ppl_}"
  PPL_VERSION="${PPL_VERSION%.tar.gz}"
  local ppl_dir="ppl_compile"
  if [ -d "${ppl_dir}" ]; then
    local current_version=""
    if [ -f "${ppl_dir}/version" ]; then
      current_version=$(cat "${ppl_dir}/version")
    fi
    if [ "${current_version}" = "${PPL_VERSION}" ]; then
      echo "PPL compiler ${current_version} already exists, skipping."
      return 0
    fi
    echo "PPL compiler ${current_version} != ${PPL_VERSION}, updating..."
    rm -rf "${ppl_dir}"
  fi
  echo "Downloading PPL compiler..."
  if [ ! -e "${PPL_PACKAGE}" ]; then
    wget "${PPL_URL}"
  fi
  # Extract PPL compiler package
  tar xvf "${PPL_PACKAGE}"
  local ppl_extracted="ppl_${PPL_VERSION}"
  mkdir -p "${ppl_dir}"
  mv "${ppl_extracted}"/* "${ppl_dir}/"
  rm -rf "${ppl_extracted}"
  rm -f "${PPL_PACKAGE}"
  chmod +x "${ppl_dir}/bin/"*
  echo "${PPL_VERSION}" > "${ppl_dir}/version"
  echo "PPL compiler downloaded to ${ppl_dir}"
}

function download_gcc_arm() {
  local tool_name="gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
  local tool_addr="https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/${tool_name}.tar.xz"
  local tool_file="${tool_name}.tar.xz"
  download_toolchain "${tool_name}" "${tool_addr}" "${tool_file}"
}

function download_gcc_linaro() {
  local tool_name="gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu"
  local tool_addr="https://releases.linaro.org/components/toolchain/binaries/6.3-2017.05/aarch64-linux-gnu/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz"
  local tool_file="${tool_name}.tar.xz"
  download_toolchain "${tool_name}" "${tool_addr}" "${tool_file}"
}

function download_riscv_xuantie900() {
  local tool_name="Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1"
  local tool_addr="https://occ-oss-prod.oss-cn-hangzhou.aliyuncs.com/resource//1695015316167/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz"
  local tool_file="Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906.tar.gz"
  download_toolchain "${tool_name}" "${tool_addr}" "${tool_file}"
}

function download_loong() {
  local tool_name="loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1"
  local tool_addr="http://ftp.loongnix.cn/toolchain/gcc/release/loongarch/gcc8/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz"
  local tool_file="loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz"
  download_toolchain "${tool_name}" "${tool_addr}" "${tool_file}"
}

# Parse arguments
TARGETS=()
CROSS_TOOLCHAINS=""

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dir)
      CROSS_TOOLCHAINS="$2"
      shift 2
      ;;
    ppl|cross-gcc|all)
      TARGETS+=("$1")
      shift
      ;;
    *)
      echo "Unknown target or option: $1"
      usage
      exit 1
      ;;
  esac
done

# Default: download everything
if [ ${#TARGETS[@]} -eq 0 ]; then
  TARGETS=("all")
fi

# Expand "all" and "cross-gcc" groups
EXPANDED_TARGETS=()
for t in "${TARGETS[@]}"; do
  case "$t" in
    all)
      EXPANDED_TARGETS+=(ppl gcc-arm gcc-linaro riscv-xuantie900 loong)
      ;;
    cross-gcc)
      EXPANDED_TARGETS+=(gcc-arm gcc-linaro riscv-xuantie900 loong)
      ;;
    *)
      EXPANDED_TARGETS+=("$t")
      ;;
  esac
done

# Deduplicate targets
UNIQUE_TARGETS=()
for t in "${EXPANDED_TARGETS[@]}"; do
  skip=false
  for u in "${UNIQUE_TARGETS[@]}"; do
    if [ "$t" = "$u" ]; then skip=true; break; fi
  done
  if ! $skip; then UNIQUE_TARGETS+=("$t"); fi
done

# Set download directory
CROSS_TOOLCHAINS=${CROSS_TOOLCHAINS:-${DIR}/cross_toolchains}

mkdir -p "${CROSS_TOOLCHAINS}"
pushd "${CROSS_TOOLCHAINS}"

for t in "${UNIQUE_TARGETS[@]}"; do
  case "$t" in
    ppl)           download_ppl ;;
    gcc-arm)       download_gcc_arm ;;
    gcc-linaro)    download_gcc_linaro ;;
    riscv-xuantie900) download_riscv_xuantie900 ;;
    loong)         download_loong ;;
  esac
done

popd
