#!/bin/bash
# Script to build TPU1686 backends for various chips and copy libs to tpu-mlir.
# Default: RELEASE build for all chips.
# Usage: build_tpu1686.sh [-d] [-c <chip>] [-t <TPU1686_PATH>]
#   -d, --debug        Build in DEBUG mode (default: RELEASE)
#   -c, --chip CHIP    Build only the specified chip (can be repeated)
#   -t, --tpu1686 PATH Path to TPU1686 directory (default: /workspace/TPU1686)
#   -h, --help         Show this help message
#
# Supported chips: bm1684x  bm1688  bm1690  bm1690e  cv184x  sgtpuv8  bm1684x2

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPUMLIR_PATH="${PROJECT_ROOT}"
LIB_DEST="${TPUMLIR_PATH}/third_party/nntoolchain/lib"
TPU1686_PATH="/workspace/TPU1686"
SHA_FILE="${SCRIPT_DIR}/tpu1686_sha256.txt"
BUILD_TYPE="RELEASE"

ALL_CHIPS=(bm1684x bm1688 bm1690 bm1690e cv184x sgtpuv8 bm1684x2)
SELECTED_CHIPS=()

# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --debug          Build in DEBUG mode (default: RELEASE)"
    echo "  -c, --chip CHIP      Build only the specified chip (repeatable)"
    echo "                       Supported: ${ALL_CHIPS[*]}"
    echo "  -t, --tpu1686 PATH   Path to TPU1686 source directory"
    echo "                       (default: /workspace/TPU1686)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")                        # all chips, RELEASE"
    echo "  $(basename "$0") -d                     # all chips, DEBUG"
    echo "  $(basename "$0") -c bm1684x             # bm1684x only, RELEASE"
    echo "  $(basename "$0") -d -c bm1688 -c bm1690 # bm1688+bm1690, DEBUG"
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug)
            BUILD_TYPE="DEBUG"
            shift ;;
        -c|--chip)
            SELECTED_CHIPS+=("$2")
            shift 2 ;;
        -t|--tpu1686)
            TPU1686_PATH="$2"
            shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Error: unknown option '$1'"
            usage ;;
    esac
done

# Default to all chips when none specified
[[ ${#SELECTED_CHIPS[@]} -eq 0 ]] && SELECTED_CHIPS=("${ALL_CHIPS[@]}")

BUILD_ALL=false
[[ ${#SELECTED_CHIPS[@]} -eq ${#ALL_CHIPS[@]} ]] && BUILD_ALL=true

# Validate chip names
for chip in "${SELECTED_CHIPS[@]}"; do
    found=false
    for c in "${ALL_CHIPS[@]}"; do [[ "$chip" == "$c" ]] && found=true && break; done
    if [[ "$found" == "false" ]]; then
        echo "Error: unsupported chip '${chip}'. Supported: ${ALL_CHIPS[*]}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Pre-flight checks
if [[ ! -d "${TPU1686_PATH}" ]]; then
    echo "Error: TPU1686 directory not found: ${TPU1686_PATH}"
    exit 1
fi
mkdir -p "${LIB_DEST}"

echo "========================================"
echo "Build type : ${BUILD_TYPE}"
echo "Chips      : ${SELECTED_CHIPS[*]}"
echo "TPU1686    : ${TPU1686_PATH}"
echo "Lib dest   : ${LIB_DEST}"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: print section header
header() { echo ""; echo ">>> [$1] starting (${BUILD_TYPE}) ..."; }

# Helper: update (or insert) a chip's git-log section in SHA_FILE without
# touching other chips' sections.
update_sha_record() {
    local chip="$1"
    local git_log
    git_log=$(git -C "${TPU1686_PATH}" log -1)
    SHA_FILE="${SHA_FILE}" CHIP="${chip}" GIT_LOG="${git_log}" python3 <<'PYEOF'
import os, re

filepath = os.environ['SHA_FILE']
chip     = os.environ['CHIP']
git_log  = os.environ['GIT_LOG']

new_section = "=== {} ===\n{}\n".format(chip, git_log)

if not os.path.exists(filepath):
    with open(filepath, 'w') as f:
        f.write(new_section)
else:
    with open(filepath, 'r') as f:
        content = f.read()
    pattern = r'=== ' + re.escape(chip) + r' ===\n.*?(?=\n=== |\Z)'
    if re.search(pattern, content, re.DOTALL):
        content = re.sub(pattern, new_section.rstrip('\n'), content, flags=re.DOTALL)
    else:
        content = (content.rstrip('\n') + '\n' + new_section) if content.strip() else new_section
    with open(filepath, 'w') as f:
        f.write(content)
PYEOF
}

# ---------------------------------------------------------------------------
# Each chip build runs in a subshell so that sourcing envsetup.sh for one
# chip does not pollute the environment of the next chip.

build_bm1684x() {
    header bm1684x
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh bm1684x
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_bm1684x.so          "${LIB_DEST}/libbackend_bm1684x.so"
        cp build_runtime/firmware_core/libcmodel_firmware.a "${LIB_DEST}/libcmodel_bm1684x.a"
        rebuild_firmware
        cp build/firmware_core/libfirmware_core.a            "${LIB_DEST}/libbm1684x_kernel_module.a"
    )
    echo ">>> [bm1684x] done."
}

build_bm1688() {
    header bm1688
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh bm1686
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_bm1686.so            "${LIB_DEST}/libbackend_bm1688.so"
        cp build_runtime/firmware_core/libcmodel_firmware.a  "${LIB_DEST}/libcmodel_bm1688.a"
        rebuild_firmware
        cp build/firmware_core/libfirmware_core.a             "${LIB_DEST}/libbmtpulv60_kernel_module.a"
    )
    echo ">>> [bm1688] done."
}

build_bm1690() {
    header bm1690
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh sg2260
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_sg2260.so "${LIB_DEST}/libbackend_bm1690.so"
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            export EXTRA_CONFIG="-DDEBUG=ON -DUSING_FW_DEBUG=ON"
        else
            export EXTRA_CONFIG="-DDEBUG=OFF -DUSING_FW_DEBUG=OFF"
        fi
        rebuild_test sgdnn
        cp build/firmware_core/libcmodel_firmware.a "${LIB_DEST}/libcmodel_bm1690.a"
        unset EXTRA_CONFIG && rebuild_firmware
        cp build/firmware_core/libfirmware_core.a   "${LIB_DEST}/libbm1690_kernel_module.a"
    )
    echo ">>> [bm1690] done."
}

build_bm1690e() {
    header bm1690e
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh sg2260e
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_sg2260e.so "${LIB_DEST}/libbackend_bm1690e.so"
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            export EXTRA_CONFIG="-DDEBUG=ON -DUSING_FW_DEBUG=ON"
        else
            export EXTRA_CONFIG="-DDEBUG=OFF -DUSING_FW_DEBUG=OFF"
        fi
        rebuild_test sgdnn
        cp build/firmware_core/libcmodel_firmware.a "${LIB_DEST}/libcmodel_bm1690e.a"
        unset EXTRA_CONFIG && rebuild_firmware
        cp build/firmware_core/libfirmware_core.a   "${LIB_DEST}/libbm1690e_kernel_module.a"
    )
    echo ">>> [bm1690e] done."
}

build_cv184x() {
    header cv184x
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh mars3
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_mars3.so             "${LIB_DEST}/libbackend_cv184x.so"
        cp build_runtime/firmware_core/libcmodel_firmware.so "${LIB_DEST}/libcmodel_cv184x.so"
    )
    echo ">>> [cv184x] done."
}

build_sgtpuv8() {
    header sgtpuv8
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh sgtpuv8
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_sgtpuv8.so           "${LIB_DEST}/libbackend_sgtpuv8.so"
        cp build_runtime/firmware_core/libcmodel_firmware.so "${LIB_DEST}/libcmodel_sgtpuv8.so"
        rebuild_firmware
        cp build/firmware_core/libfirmware_core.so "${LIB_DEST}/libsgtpuv8_kernel_module.so"
        cp build/firmware_core/libfirmware_core.a  "${LIB_DEST}/libsgtpuv8_kernel_module.a"
    )
    echo ">>> [sgtpuv8] done."
}

build_bm1684x2() {
    header bm1684x2
    (
        cd "${TPU1686_PATH}"
        source scripts/envsetup.sh bm1684x2
        if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
            rebuild_backend_lib_cmodel
        else
            unset EXTRA_CONFIG && rebuild_backend_lib_release_cmodel
        fi
        cp build/backend_api/libbackend_bm1684x2.so          "${LIB_DEST}/libbackend_bm1684x2.so"
        cp build_runtime/firmware_core/libcmodel_firmware.a  "${LIB_DEST}/libcmodel_bm1684x2.a"
        rebuild_firmware
        cp build/firmware_core/libfirmware_core.a            "${LIB_DEST}/libbm1684x2_kernel_module.a"
    )
    echo ">>> [bm1684x2] done."
}

# ---------------------------------------------------------------------------
# Dispatch
for chip in "${SELECTED_CHIPS[@]}"; do
    case "${chip}" in
        bm1684x)  build_bm1684x  ;;
        bm1688)   build_bm1688   ;;
        bm1690)   build_bm1690   ;;
        bm1690e)  build_bm1690e  ;;
        cv184x)   build_cv184x   ;;
        sgtpuv8)  build_sgtpuv8  ;;
        bm1684x2) build_bm1684x2 ;;
    esac
    update_sha_record "${chip}"
done

# ---------------------------------------------------------------------------
# Build PplBackend once after all backends are ready
# echo ""
# echo ">>> [PplBackend] building ..."
# if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
#     "${TPUMLIR_PATH}/lib/PplBackend/build.sh" DEBUG
# else
#     "${TPUMLIR_PATH}/lib/PplBackend/build.sh" RELEASE
# fi
# echo ">>> [PplBackend] done."

echo ""
echo "========================================"
echo "All done."
echo "========================================"
