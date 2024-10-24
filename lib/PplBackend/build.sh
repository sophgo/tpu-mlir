#!/bin/bash

set -ex

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
pushd "$DIR"

# clear cache
CACHE_PATH=${PPL_CACHE_PATH}
if [ x${CACHE_PATH} = x ];then
  CACHE_PATH=${HOME}"/.ppl/cache"
fi

rm -rf $CACHE_PATH

mkdir -p build
cd build

for file in `ls ../src/*.pl`
do
  ppl-compile $file --I $PPL_PROJECT_ROOT/inc  --desc --O2 --o .
done

cmake ../ -DDEBUG=ON -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}"
make install

popd
