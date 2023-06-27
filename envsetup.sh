#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
export BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}
export TPUC_ROOT=$INSTALL_PATH

echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "BUILD_PATH   : ${BUILD_PATH}"
echo "INSTALL_PATH : ${INSTALL_PATH}"

# regression path
export REGRESSION_PATH=$PROJECT_ROOT/regression
export NNMODELS_PATH=${PROJECT_ROOT}/../nnmodels
export MODEL_ZOO_PATH=${PROJECT_ROOT}/../model-zoo

# run path
export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$PROJECT_ROOT/llvm/bin:$PATH
export PATH=$PROJECT_ROOT/python/tools:$PATH
export PATH=$PROJECT_ROOT/python/utils:$PATH
export PATH=$PROJECT_ROOT/python/test:$PATH
export PATH=$PROJECT_ROOT/python/samples:$PATH
export PATH=$PROJECT_ROOT/third_party/customlayer/python:$PATH

export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$PROJECT_ROOT/capi/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$INSTALL_PATH/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/llvm/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/caffe/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/customlayer/python:$PYTHONPATH

export OMP_NUM_THREADS=4
