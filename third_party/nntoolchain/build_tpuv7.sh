#!/bin/bash
# Script to build tpuv7-runtime (emulator) and copy libs/headers to tpu-mlir.
# Default: RELEASE build.
# Usage: build_tpuv7.sh [-d] [-t <TPUV7_PATH>]
#   -d, --debug        Build in DEBUG mode (default: RELEASE)
#   -t, --tpuv7 PATH   Path to tpuv7-runtime directory (default: /workspace/tpuv7-runtime)
#   -h, --help         Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TPUMLIR_PATH="${PROJECT_ROOT}"
LIB_DEST="${TPUMLIR_PATH}/third_party/nntoolchain/lib"
INC_DEST="${TPUMLIR_PATH}/third_party/nntoolchain/include"
TPUV7_PATH="/workspace/tpuv7-runtime"
SHA_FILE="${SCRIPT_DIR}/tpuv7_sha256.txt"
BUILD_TYPE="RELEASE"

# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --debug          Build in DEBUG mode (default: RELEASE)"
    echo "  -t, --tpuv7 PATH     Path to tpuv7-runtime source directory"
    echo "                       (default: /workspace/tpuv7-runtime)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $(basename "$0")                    # RELEASE build"
    echo "  $(basename "$0") -d                 # DEBUG build"
    echo "  $(basename "$0") -t /path/to/tpuv7  # custom path, RELEASE"
    exit 0
}

# ---------------------------------------------------------------------------
# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug)
            BUILD_TYPE="DEBUG"
            shift ;;
        -t|--tpuv7)
            TPUV7_PATH="$2"
            shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Error: unknown option '$1'"
            usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-flight checks
if [[ ! -d "${TPUV7_PATH}" ]]; then
    echo "Error: tpuv7-runtime directory not found: ${TPUV7_PATH}"
    exit 1
fi
mkdir -p "${LIB_DEST}" "${INC_DEST}"

if [[ "${BUILD_TYPE}" == "DEBUG" ]]; then
    USING_DEBUG="ON"
    USING_TP_DEBUG="ON"
else
    USING_DEBUG="OFF"
    USING_TP_DEBUG="OFF"
fi

echo "========================================"
echo "Build type  : ${BUILD_TYPE}"
echo "tpuv7-runtime: ${TPUV7_PATH}"
echo "Lib dest    : ${LIB_DEST}"
echo "Include dest: ${INC_DEST}"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
BUILD_DIR="${TPUV7_PATH}/build/emulator"

echo ">>> [tpuv7-runtime] building (${BUILD_TYPE}) ..."
mkdir -p "${BUILD_DIR}"
pushd "${BUILD_DIR}"

cmake \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/../install" \
    -DUSING_CMODEL=ON \
    -DUSING_DEBUG="${USING_DEBUG}" \
    -DUSING_TP_DEBUG="${USING_TP_DEBUG}" \
    ../..

make -j"$(nproc)"

popd

# ---------------------------------------------------------------------------
echo ">>> [tpuv7-runtime] copying libraries ..."
cp "${BUILD_DIR}/model-runtime/runtime/libtpuv7_modelrt.so"             "${LIB_DEST}/libtpuv7_modelrt.so"
cp "${BUILD_DIR}/cdmlib/host/cdm_runtime/libtpuv7_rt.so"                "${LIB_DEST}/libtpuv7_rt.so"
cp "${BUILD_DIR}/cdmlib/fw/ap/daemon/libcdm_daemon_emulator.so"         "${LIB_DEST}/libcdm_daemon_emulator.so"
cp "${BUILD_DIR}/cdmlib/fw/tp/daemon/libtpuv7_scalar_emulator.so"       "${LIB_DEST}/libtpuv7_scalar_emulator.so"

echo ">>> [tpuv7-runtime] copying headers ..."
cp "${TPUV7_PATH}/model-runtime/runtime/include/tpuv7_modelrt.h"        "${INC_DEST}/tpuv7_modelrt.h"
cp "${TPUV7_PATH}/cdmlib/host/cdm_runtime/include/tpuv7_rt.h"           "${INC_DEST}/tpuv7_rt.h"

# ---------------------------------------------------------------------------
# Record git log to SHA file
echo ">>> [tpuv7-runtime] updating ${SHA_FILE} ..."
GIT_LOG=$(git -C "${TPUV7_PATH}" log -1)
SHA_FILE="${SHA_FILE}" GIT_LOG="${GIT_LOG}" python3 <<'PYEOF'
import os

filepath = os.environ['SHA_FILE']
git_log  = os.environ['GIT_LOG']

new_content = "=== tpuv7-runtime ===\n{}\n".format(git_log)

with open(filepath, 'w') as f:
    f.write(new_content)
PYEOF

echo ""
echo "========================================"
echo "tpuv7-runtime build done."
echo "========================================"
