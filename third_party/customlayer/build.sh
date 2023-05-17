#!/bin/bash

rm -rf build
mkdir build
cd build

cmake -DCMAKE_INSTALL_PREFIX="${TPUC_ROOT}" ..

make

make install
