#!/bin/bash
set -e

source envsetup.sh
rm -rf ${INSTALL_PATH}
rm -rf ${PROJECT_ROOT}/regression/regression_out
source build.sh RELEASE

export mlir_version="$(grep MLIR_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
release_archive="./tpu_mlir"

rm -rf ${release_archive}
mkdir -p ${release_archive}
cp -rf ${INSTALL_PATH}/* ${release_archive}

cp -rf ${PROJECT_ROOT}/regression ${release_archive}
rm -rf ${release_archive}/regression/model
cp -rf ${PROJECT_ROOT}/third_party/customlayer ${release_archive}
cp -rf /usr/local/python_packages/caffe/ ${release_archive}/python/caffe/
cp ${PROJECT_ROOT}/release_tools/{__init__.py,entryconfig.py} ${release_archive}
cp ${PROJECT_ROOT}/release_tools/{setup.py,MANIFEST.in} ${PROJECT_ROOT}
touch ${release_archive}/__version__
echo ${mlir_version} > ${release_archive}/__version__

# collect_caffe_dependence
mkdir -p ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libboost_thread.so.1.74.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libboost_regex.so.1.74.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libglog.so.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libgflags.so.2.2 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libprotobuf.so.23 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libm.so.6 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libboost_python310.so.1.74.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libgomp.so.1 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libz.so.1 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libicui18n.so.70 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libicuuc.so.70 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libunwind.so.8 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libpthread.so.0  ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libgfortran.so.5  ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/liblzma.so.5 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libquadmath.so.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libopenblas.so.0  ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libgcc_s.so.1  ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libicudata.so.70  ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libc.so.6 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libomp.so.5 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libglib-2.0.so.0 ${release_archive}/lib/third_party/
cp /usr/lib/x86_64-linux-gnu/libgthread-2.0.so.0 ${release_archive}/lib/third_party/

# collect_oneDNN_dependence
cp /usr/local/lib/libdnnl.so ${release_archive}/lib/third_party/
cp /usr/local/lib/libdnnl.so.3 ${release_archive}/lib/third_party/
cp /usr/local/lib/libdnnl.so.3.1 ${release_archive}/lib/third_party/

# collect_capi_dependence
cp -rf ${PROJECT_ROOT}/capi/lib/* ${release_archive}/lib/third_party/

# automic entries gen for entry.py and set for setup.py
python ${release_archive}/entryconfig.py bin/ python/tools/ python/samples/

# build pip package
python -m build

# clean files
rm dist/*.tar.gz
rm -rf ${release_archive} && rm -rf ${release_archive}.egg-info
rm setup.py && rm MANIFEST.in