#!/bin/bash
package_name="tpu_mlir"

package_info=$(pip show $package_name)
package_path=$(echo "$package_info" | awk -F ': ' '/^Location:/ {print $2}')
RELEASE_PATH="${package_path}/${package_name}"

export PROJECT_ROOT=$RELEASE_PATH
export TPUC_ROOT=$RELEASE_PATH

# echo "PROJECT_ROOT : ${PROJECT_ROOT}"
echo "RELEASE_PATH   : ${RELEASE_PATH}"

# regression path
export REGRESSION_PATH=$PROJECT_ROOT/regression
export NNMODELS_PATH=${PROJECT_ROOT}/../nnmodels
export MODEL_ZOO_PATH=${PROJECT_ROOT}/../model-zoo

# release package path
export PATH=$PROJECT_ROOT/bin:$PATH
export PATH=$PROJECT_ROOT/python/tools:$PATH
export PATH=$PROJECT_ROOT/python/utils:$PATH
export PATH=$PROJECT_ROOT/python/test:$PATH
export PATH=$PROJECT_ROOT/python/samples:$PATH
export PATH=$PROJECT_ROOT/customlayer/python:$PATH

export LD_LIBRARY_PATH=$RELEASE_PATH/lib:$RELEASE_PATH/lib/third_party/:$PROJECT_ROOT/capi/lib:$LD_LIBRARY_PATH

export PYTHONPATH=$RELEASE_PATH/:$PYTHONPATH
export PYTHONPATH=$RELEASE_PATH/python/:$PYTHONPATH
export PYTHONPATH=$RELEASE_PATH/lib/:$PYTHONPATH