#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
export BUILD_PATH=${BUILD_PATH:-$PROJECT_ROOT/build}
export INSTALL_PATH=${INSTALL_PATH:-$PROJECT_ROOT/install}

echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "BUILD_PATH   : ${BUILD_PATH}"
echo "INSTALL_PATH : ${INSTALL_PATH}"

# regression path
export REGRESSION_PATH=$PROJECT_ROOT/regression
# run path
export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$PROJECT_ROOT/llvm/bin:$PATH
export PATH=$PROJECT_ROOT/python/tools:$PATH

export LD_LIBRARY_PATH=$INSTALL_PATH/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$INSTALL_PATH/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/llvm/python_packages/mlir_core:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH

export OMP_NUM_THREADS=4
