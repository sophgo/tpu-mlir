#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export PROJECT_ROOT=$DIR
echo "PROJECT_ROOT : ${PROJECT_ROOT}"

# run path
export PATH=$PROJECT_ROOT:$PATH
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/debugger
export USING_CMODEL=False
export LD_LIBRARY_PATH=$PROJECT_ROOT/debugger/lib:$LD_LIBRARY_PATH