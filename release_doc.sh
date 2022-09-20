#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# generate tpu-mlir_quick_start
# ------------------------------------------------------------------------------
pushd ${DIR}/doc/quick_start
make clean
make pdf
make html
popd

# generate tpu-mlir_technical_manual
# ------------------------------------------------------------------------------
pushd ${DIR}/doc/developer_manual
make clean
make pdf
make html
popd
# ------------------------------------------------------------------------------

mkdir -p ${DIR}/build/doc/quick_start_zh
cp -rf ${DIR}/doc/quick_start/build_zh/tpu-mlir_quick_start_zh.pdf \
   ${DIR}/build/doc/quick_start_zh/
cp -rf ${DIR}/doc/quick_start/build_zh/html ${DIR}/build/doc/quick_start_zh

mkdir -p ${DIR}/build/doc/quick_start_en
cp -rf ${DIR}/doc/quick_start/build_en/tpu-mlir_quick_start_en.pdf \
   ${DIR}/build/doc/quick_start_en/
cp -rf ${DIR}/doc/quick_start/build_en/html ${DIR}/build/doc/quick_start_en

mkdir -p ${DIR}/build/doc/developer_manual_zh
cp -f ${DIR}/doc/developer_manual/build_zh/tpu-mlir_technical_manual_zh.pdf \
   ${DIR}/build/doc/developer_manual_zh/
cp -rf ${DIR}/doc/developer_manual/build_zh/html ${DIR}/build/doc/developer_manual_zh/

mkdir -p ${DIR}/build/doc/developer_manual_en
cp -f ${DIR}/doc/developer_manual/build_en/tpu-mlir_technical_manual_en.pdf \
   ${DIR}/build/doc/developer_manual_en/
cp -rf ${DIR}/doc/developer_manual/build_en/html ${DIR}/build/doc/developer_manual_en/
