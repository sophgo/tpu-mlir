#!/bin/bash
set -e

######################### build tpu_mlir core package #######################
# build RELEASE
source envsetup.sh
rm -rf ${INSTALL_PATH}
rm -rf ${PROJECT_ROOT}/regression/regression_out
source build.sh RELEASE

# set mlir_version
export mlir_version="$(grep MLIR_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"

# collect tpu_mlir core files
export release_archive="./tpu_mlir"
rm -rf ${release_archive}
mkdir -p ${release_archive}
cp -rf ${INSTALL_PATH}/* ${release_archive}
rm ${release_archive}/python/mlir/_mlir_libs/libTPUMLIRPythonCAPI.so
cp -rf ${PROJECT_ROOT}/regression ${release_archive}
rm -rf ${release_archive}/regression/model
cp -rf ${PROJECT_ROOT}/third_party/customlayer ${release_archive}

mkdir -p ${release_archive}/python/
cp -rf /usr/local/python_packages/caffe/ ${release_archive}/python/caffe/
cp ${PROJECT_ROOT}/release_tools/{__init__.py,entryconfig.py} ${release_archive}
cp ${PROJECT_ROOT}/release_tools/{setup.py,MANIFEST.in} ${PROJECT_ROOT}

# collect_caffe_dependence
mkdir -p ${release_archive}/lib/third_party/

for file in libpython3.10.so.1.0 \
    libboost_thread.so.1.74.0 \
    libboost_filesystem.so.1.74.0 \
    libboost_regex.so.1.74.0 \
    libglog.so.0 \
    libgflags.so.2.2 \
    libprotobuf.so.23 \
    libm.so.6 \
    libboost_python310.so.1.74.0 \
    libstdc++.so.6 \
    libgomp.so.1 \
    libz.so.1 \
    libicui18n.so.70 \
    libicuuc.so.70 \
    libunwind.so.8 \
    libpthread.so.0 \
    libgfortran.so.5 \
    liblzma.so.5 \
    libquadmath.so.0 \
    libopenblas.so.0 \
    libgcc_s.so.1 \
    libicudata.so.70 \
    libc.so.6 \
    libomp.so.5 \
    libglib-2.0.so.0 \
    libgthread-2.0.so.0 \
    libGL.so.1 \
    libGLdispatch.so.0 \
    libGLX.so.0 \
    libtinfo.so.5
do
    cp /usr/lib/x86_64-linux-gnu/${file} ${release_archive}/lib/third_party/
done

# collect_oneDNN_dependence
for file in {libdnnl.so,libdnnl.so.3,libdnnl.so.3.1}
do
    cp -d /usr/local/lib/${file} ${release_archive}/lib/third_party/
done

# collect_capi_dependence
cp -rf ${PROJECT_ROOT}/capi/lib/* ${release_archive}/lib/third_party/

# automic entries gen for entry.py and set for setup.py
python ${release_archive}/entryconfig.py bin/ python/tools/ python/samples/

# set tpu-mlir core shared object files rpath
patchelf --set-rpath '$ORIGIN/../lib/:$ORIGIN/../lib/third_party/' ${release_archive}/bin/cvimodel_debug
patchelf --set-rpath '$ORIGIN/../lib/:$ORIGIN/../lib/third_party/' ${release_archive}/bin/model_tool
patchelf --set-rpath '$ORIGIN/../lib/:$ORIGIN/../lib/third_party/' ${release_archive}/bin/tpuc-opt
patchelf --set-rpath '$ORIGIN/../lib/:$ORIGIN/../lib/third_party/' ${release_archive}/python/pymlir.cpython-310-x86_64-linux-gnu.so
patchelf --set-rpath '$ORIGIN/../lib/:$ORIGIN/../lib/third_party/' ${release_archive}/python/pyruntime_bm.cpython-310-x86_64-linux-gnu.so
patchelf --set-rpath '$ORIGIN/../lib/:$ORIGIN/../lib/third_party/' ${release_archive}/python/pyruntime_cvi.cpython-310-x86_64-linux-gnu.so
patchelf --set-rpath '$ORIGIN/../../lib/:$ORIGIN/../../lib/third_party/' ${release_archive}/python/caffe/_caffe.so
patchelf --set-rpath '$ORIGIN/../../../lib/:$ORIGIN/../../../lib/third_party/' ${release_archive}/python/mlir/_mlir_libs/libTPUMLIRPythonCAPI.so.18git
patchelf --set-rpath '$ORIGIN/../../../lib/:$ORIGIN/../../../lib/third_party/:$ORIGIN/' ${release_archive}/python/mlir/_mlir_libs/_mlir.cpython-310-x86_64-linux-gnu.so
patchelf --set-rpath '$ORIGIN/../../../lib/:$ORIGIN/../../../lib/third_party/:$ORIGIN/' ${release_archive}/python/mlir/_mlir_libs/_mlirDialectsQuant.cpython-310-x86_64-linux-gnu.so
patchelf --set-rpath '$ORIGIN/../../../lib/:$ORIGIN/../../../lib/third_party/:$ORIGIN/' ${release_archive}/python/mlir/_mlir_libs/_mlirRegisterEverything.cpython-310-x86_64-linux-gnu.so
for file in ${release_archive}/lib/*
do
    if [ -f "$file" ]
    then
        patchelf --set-rpath '$ORIGIN/:$ORIGIN/../lib/third_party/' $file
    fi
done

# set tpu-mlir dependence shared object files rpath
for file in ${release_archive}/lib/third_party/*
do
    if [ -f "$file" ]
    then
        patchelf --set-rpath '$ORIGIN/../:$ORIGIN/' $file
    fi
done

# build pip package
python -m build

# clean files
rm -rf ${release_archive} && rm -rf ${release_archive}.egg-info
rm setup.py && rm MANIFEST.in