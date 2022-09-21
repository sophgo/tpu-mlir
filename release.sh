#!/bin/bash
set -e

source envsetup.sh
rm -rf ${INSTALL_PATH}
rm -rf ${PROJECT_ROOT}/regression/regression_out
source build.sh RELEASE

mlir_version="$(grep MLIR_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
release_archive="./tpu-mlir_${mlir_version}"

rm -rf ${release_archive}*
cp -rf ${INSTALL_PATH} ${release_archive}

cp -rf ${PROJECT_ROOT}/regression ${release_archive}

# generate tpu-mlir_quick_start_guide
# ------------------------------------------------------------------------------
pushd ${PROJECT_ROOT}/doc/quick_start
make clean
make pdf
popd
mkdir -p ${release_archive}/docs
cp -f ${PROJECT_ROOT}/doc/quick_start/build_zh/tpu-mlir_quick_start_zh.pdf \
   ${release_archive}/docs/"TPU-MLIR快速入门指南.pdf"
cp -f ${PROJECT_ROOT}/doc/quick_start/build_en/tpu-mlir_quick_start_en.pdf \
   ${release_archive}/docs/"TPU-MLIR_Quick_Start.pdf"
# ------------------------------------------------------------------------------

# generate tpu-mlir_technical_manual
# ------------------------------------------------------------------------------
pushd ${PROJECT_ROOT}/doc/developer_manual
make clean
make pdf
popd
cp -f ${PROJECT_ROOT}/doc/developer_manual/build_zh/tpu-mlir_technical_manual_zh.pdf \
   ${release_archive}/docs/"TPU-MLIR开发参考手册.pdf"
cp -f ${PROJECT_ROOT}/doc/developer_manual/build_en/tpu-mlir_technical_manual_en.pdf \
   ${release_archive}/docs/"TPU-MLIR_Technical_Reference_Manual.pdf"

# ------------------------------------------------------------------------------

# build a envsetup.sh
# ------------------------------------------------------------------------------
__envsetupfile=${release_archive}/envsetup.sh
rm -f __envsetupfile

echo "Create ${__envsetupfile}" 1>&2
more > "${__envsetupfile}" <<'//MY_CODE_STREAM'
#!/bin/bash
# set environment variable
TPUC_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PATH=${TPUC_ROOT}/bin:$PATH
export PATH=${TPUC_ROOT}/python/tools:$PATH
export PATH=${TPUC_ROOT}/python/utils:$PATH
export PATH=${TPUC_ROOT}/python/test:$PATH
export PATH=${TPUC_ROOT}/python/samples:$PATH
export LD_LIBRARY_PATH=$TPUC_ROOT/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${TPUC_ROOT}/python:$PYTHONPATH
export MODEL_ZOO_PATH=${TPUC_ROOT}/../model-zoo
//MY_CODE_STREAM
# ------------------------------------------------------------------------------


tar -cvzf "tpu-mlir_${mlir_version}.tar.gz" ${release_archive}
rm -rf ${release_archive}
