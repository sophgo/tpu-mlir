#!/bin/bash
set -ex

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rm -rf regression_out
mkdir regression_out
pushd regression_out

# run basic regression
$DIR/basic/run.sh

# run nnmodels regression if exist nnmodels
$DIR/nnmodels/run.sh

popd
