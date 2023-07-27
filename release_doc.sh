#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python docs/generate_operation.py ${DIR}/build/supported_ops.rst

# generate tpu-mlir_quick_start
# ------------------------------------------------------------------------------
pushd ${DIR}/docs/quick_start
make clean
make pdf
make html
popd

# generate tpu-mlir_technical_manual
# ------------------------------------------------------------------------------
pushd ${DIR}/docs/developer_manual
make clean
make pdf
make html
popd
# ------------------------------------------------------------------------------

mkdir -p ${DIR}/build/docs/quick_start_zh
cp -rf ${DIR}/docs/quick_start/build_zh/tpu-mlir_quick_start_zh.pdf \
   ${DIR}/build/docs/quick_start_zh/
cp -rf ${DIR}/docs/quick_start/build_zh/html ${DIR}/build/docs/quick_start_zh

mkdir -p ${DIR}/build/docs/quick_start_en
cp -rf ${DIR}/docs/quick_start/build_en/tpu-mlir_quick_start_en.pdf \
   ${DIR}/build/docs/quick_start_en/
cp -rf ${DIR}/docs/quick_start/build_en/html ${DIR}/build/docs/quick_start_en

mkdir -p ${DIR}/build/docs/developer_manual_zh
cp -f ${DIR}/docs/developer_manual/build_zh/tpu-mlir_technical_manual_zh.pdf \
   ${DIR}/build/docs/developer_manual_zh/
cp -rf ${DIR}/docs/developer_manual/build_zh/html ${DIR}/build/docs/developer_manual_zh/

mkdir -p ${DIR}/build/docs/developer_manual_en
cp -f ${DIR}/docs/developer_manual/build_en/tpu-mlir_technical_manual_en.pdf \
   ${DIR}/build/docs/developer_manual_en/
cp -rf ${DIR}/docs/developer_manual/build_en/html ${DIR}/build/docs/developer_manual_en/

if [[ ! -z "$INSTALL_PATH" ]]; then
# only install pdf
mkdir -p ${INSTALL_PATH}/docs
cp -f ${DIR}/docs/quick_start/build_zh/tpu-mlir_quick_start_zh.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR快速入门指南.pdf"
cp -f ${PROJECT_ROOT}/docs/quick_start/build_en/tpu-mlir_quick_start_en.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR_Quick_Start.pdf"
cp -f ${PROJECT_ROOT}/docs/developer_manual/build_zh/tpu-mlir_technical_manual_zh.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR开发参考手册.pdf"
cp -f ${PROJECT_ROOT}/docs/developer_manual/build_en/tpu-mlir_technical_manual_en.pdf \
   ${INSTALL_PATH}/docs/"TPU-MLIR_Technical_Reference_Manual.pdf"
fi
