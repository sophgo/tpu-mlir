#!/bin/bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
# PPL compiler version and download URL
# Update these when a new PPL release is needed
PPL_URL="/toolchains/tpu_mlir/ppl_v1.7.111-g23cf8831-20260616.tar.gz"

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

# Print the top-level directory name of a tarball, or return 1 if it cannot be
# determined reliably (e.g. corrupt archive, or first entry is "./"). Strips a
# leading "./" prefix so archives packaged that way are handled correctly.
DFSS_INSTALLED=0
function ensure_dfss() {
  if [ "${DFSS_INSTALLED}" -eq 0 ]; then
    pip3 install -U dfss
    DFSS_INSTALLED=1
  fi
}

function tar_toplevel() {
  local tarball="$1"
  local first top
  first=$(tar tf "${tarball}" 2>/dev/null | head -1)
  [ -n "${first}" ] || return 1
  first="${first#./}"
  top="${first%%/*}"
  [ -n "${top}" ] && [ "${top}" != "." ] || return 1
  printf '%s\n' "${top}"
}

function download_toolchain() {
  local addr=$1
  local filename="${addr##*/}"
  local toolchain=""
  if [ -e "${filename}" ]; then
    toolchain=$(tar_toplevel "${filename}") || true
  fi
  if [ -n "${toolchain}" ] && [ -d "${toolchain}" ]; then
    echo "${toolchain} already exists, skipping."
    return 0
  fi
  if [ ! -e "${filename}" ]; then
    echo "Downloading ${filename}..."
    ensure_dfss
    python3 -m dfss --url="open@sophgo.com:${addr}"
  else
    echo "Extracting ${filename}..."
  fi
  tar xvf "${filename}"
}

function download_ppl() {
  local ppl_package="${PPL_URL##*/}"
  local ppl_version="${ppl_package#ppl_}"
  ppl_version="${ppl_version%.tar.gz}"
  local ppl_dir="ppl_compile"
  if [ -d "${ppl_dir}" ]; then
    local current_version=""
    if [ -f "${ppl_dir}/version" ]; then
      current_version=$(cat "${ppl_dir}/version")
    fi
    if [ "${current_version}" = "${ppl_version}" ]; then
      echo "PPL compiler ${current_version} already exists, skipping."
      return 0
    fi
    echo "PPL compiler ${current_version} != ${ppl_version}, updating..."
    rm -rf "${ppl_dir}"
  fi
  echo "Downloading PPL compiler..."
  if [ ! -e "${ppl_package}" ]; then
    ensure_dfss
    python3 -m dfss --url="open@sophgo.com:${PPL_URL}"
  fi
  # Extract PPL compiler package
  tar xvf "${ppl_package}"
  local ppl_extracted=""
  ppl_extracted=$(tar_toplevel "${ppl_package}") || {
    echo "ERROR: cannot determine top-level dir of ${ppl_package}" >&2
    exit 1
  }
  if [ ! -d "${ppl_extracted}" ]; then
    echo "ERROR: expected extracted dir '${ppl_extracted}' not found" >&2
    exit 1
  fi
  mkdir -p "${ppl_dir}"
  mv "${ppl_extracted}"/* "${ppl_dir}/"
  rm -rf "${ppl_extracted}"
  rm -f "${ppl_package}"
  shopt -s nullglob
  chmod +x "${ppl_dir}/bin/"* 2>/dev/null || true
  shopt -u nullglob
  echo "${ppl_version}" > "${ppl_dir}/version"
  echo "PPL compiler downloaded to ${ppl_dir}"
}

function download_gcc_arm() {
  local url="/toolchains/tpu_mlir/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu-stripped.tar.xz"
  download_toolchain "${url}"
}

function download_gcc_linaro() {
  local url="/toolchains/tpu_mlir/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu-stripped.tar.xz"
  download_toolchain "${url}"
}

function download_riscv_xuantie900() {
  local url="/toolchains/tpu_mlir/Xuantie-900-gcc-linux-5.10.4-glibc-x86_64-V2.6.1-20220906-stripped.tar.xz"
  download_toolchain "${url}"
}

function download_loong() {
  local url="/toolchains/tpu_mlir/loongson-gnu-toolchain-8.3-x86_64-loongarch64-linux-gnu-rc1.1.tar.xz"
  download_toolchain "${url}"
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
pushd "${CROSS_TOOLCHAINS}" >/dev/null

for t in "${UNIQUE_TARGETS[@]}"; do
  case "$t" in
    ppl)           download_ppl ;;
    gcc-arm)       download_gcc_arm ;;
    gcc-linaro)    download_gcc_linaro ;;
    riscv-xuantie900) download_riscv_xuantie900 ;;
    loong)         download_loong ;;
  esac
done

popd >/dev/null
