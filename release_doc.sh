#!/bin/bash
set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# generate tpu-mlir_quick_start_guide
# ------------------------------------------------------------------------------
pushd ${DIR}/doc/quick_start
make clean
make pdf
make html
popd

# generate tpu-mlir_developer_manual
# ------------------------------------------------------------------------------
pushd ${DIR}/doc/developer_manual
make clean
make pdf
make html
popd
# ------------------------------------------------------------------------------

mkdir -p ${DIR}/build/doc/quick_start
cp -rf ${DIR}/doc/quick_start/build/tpu-mlir_quick_start_guide_zh.pdf \
   ${DIR}/build/doc/quick_start/
cp -rf ${DIR}/doc/quick_start/build/html ${DIR}/build/doc/quick_start/

mkdir -p ${DIR}/build/doc/developer_manual
cp -f ${DIR}/doc/developer_manual/build/tpu-mlir_developer_manual_zh.pdf \
   ${DIR}/build/doc/developer_manual/
cp -rf ${DIR}/doc/developer_manual/build/html ${DIR}/build/doc/developer_manual/
