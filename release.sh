#!/bin/bash
set -e

source envsetup.sh
rm -rf ${INSTALL_PATH}
rm -rf ${PROJECT_ROOT}/regression/regression_out
if [ x"$1" = x"CUDA" ]; then
    source build.sh RELEASE CUDA
else
    source build.sh RELEASE
fi

mlir_version="$(grep MLIR_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
release_archive="./tpu-mlir_${mlir_version}"

rm -rf ${release_archive}*
cp -rf ${INSTALL_PATH} ${release_archive}

cp -rf ${PROJECT_ROOT}/regression ${release_archive}
rm -rf ${release_archive}/regression/model
cp -rf ${PROJECT_ROOT}/third_party/customlayer ${release_archive}
cp -rf ${PROJECT_ROOT}/python/tools/soc_infer ${release_archive}/python/tools/

# ------------------------------------------------------------------------------

# build a envsetup.sh
# ------------------------------------------------------------------------------
__envsetupfile=${release_archive}/envsetup.sh
rm -f __envsetupfile

echo "Create ${__envsetupfile}" 1>&2
more > "${__envsetupfile}" <<'//MY_CODE_STREAM'
#!/bin/bash
# Check for Ubuntu22.04 and python3.10
os_version=$(grep "VERSION_ID" /etc/os-release | cut -d'=' -f2 | tr -d '"')
python_version=$(python3.10 -V 2>&1 | awk '{print $2}')

if [ "$os_version" != "22.04" ] || [[ ! "$python_version" == 3.10* ]]; then
    echo "Error: System requirements not met (ubuntu==22.04 and python==3.10)."
    echo "       You can use sophgo/tpuc_dev:v3.4 docker image."
    echo " hint: docker pull sophgo/tpuc_dev:v3.4"
    exit 1
fi

# set environment variable

export TPUC_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PATH=${TPUC_ROOT}/bin:$PATH
export PATH=${TPUC_ROOT}/python/tools:$PATH
export PATH=${TPUC_ROOT}/python/utils:$PATH
export PATH=${TPUC_ROOT}/python/test:$PATH
export PATH=${TPUC_ROOT}/python/samples:$PATH
export PATH=${TPUC_ROOT}/customlayer/python:$PATH
# export LD_LIBRARY_PATH=$TPUC_ROOT/lib:$LD_LIBRARY_PATH
export PPL_PROJECT_ROOT=${TPUC_ROOT}/ppl
export PPL_BUILD_PATH=$PPL_PROJECT_ROOT/build
export PPL_INSTALL_PATH=$PPL_PROJECT_ROOT/install
export PPL_RUNTIME_PATH=$PPL_PROJECT_ROOT/runtime
export PPL_THIRD_PARTY_PATH=$PPL_PROJECT_ROOT/third_party
export PATH=$PPL_PROJECT_ROOT/bin:$PATH
export USING_CMODEL=True
export PYTHONPATH=${TPUC_ROOT}/python:$PYTHONPATH
export PYTHONPATH=/usr/local/python_packages/:$PYTHONPATH
export PYTHONPATH=${TPUC_ROOT}/customlayer/python:$PYTHONPATH
export MODEL_ZOO_PATH=${TPUC_ROOT}/../model-zoo
export REGRESSION_PATH=${TPUC_ROOT}/regression
export CUSTOM_LAYER_PATH=${TPUC_ROOT}/customlayer

export CMODEL_LD_LIBRARY_PATH=$TPUC_ROOT/lib:$LD_LIBRARY_PATH
export CHIP_LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib/:$TPUC_ROOT/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CMODEL_LD_LIBRARY_PATH
function use_cmodel(){
    export USING_CMODEL=True
    export LD_LIBRARY_PATH=$CMODEL_LD_LIBRARY_PATH
}
function use_chip(){
    export USING_CMODEL=False
    export LD_LIBRARY_PATH=$CHIP_LD_LIBRARY_PATH
}
//MY_CODE_STREAM
# ------------------------------------------------------------------------------

tar -cvzf "tpu-mlir_${mlir_version}.tar.gz" ${release_archive}
rm -rf ${release_archive}
