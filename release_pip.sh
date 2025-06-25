#!/bin/bash
set -e

######################### build tpu_mlir core package #######################
# build RELEASE
source envsetup.sh
rm -rf ${INSTALL_PATH}
rm -rf ${PROJECT_ROOT}/regression/regression_out

USE_CUDA=""
TEST_MODE=""
for arg in "$@"; do
    case $arg in
        CUDA)
            USE_CUDA="CUDA"
            ;;
        test)
            TEST_MODE="test"
            ;;
        *)
            echo "Invalid option: $arg"
            exit 1
            ;;
    esac
done

# Check for CUDA support
if [ "$USE_CUDA" == "CUDA" ]; then
    echo "Building with CUDA support."
    source build.sh RELEASE $USE_CUDA
elif [ "$USE_CUDA" == "" ]; then
    echo "Building without CUDA support."
    source build.sh RELEASE
fi

# build customlayer for regression test
if [ "$TEST_MODE" == "test" ]; then
    source ${PROJECT_ROOT}/third_party/customlayer/envsetup.sh
    rebuild_custom_plugin
    rebuild_custom_backend
    rebuild_custom_firmware_cmodel bm1684x
    rebuild_custom_firmware_cmodel bm1688
    rebuild_custom_firmware_soc bm1684x
    rebuild_custom_firmware_soc bm1688
    rebuild_custom_firmware_pcie
    rm -rf ${PROJECT_ROOT}/third_party/customlayer/build
fi

# set mlir_version
export mlir_version="$(grep MLIR_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
export mlir_commit_id="$(git rev-parse --short HEAD)"
export mlir_commit_date="$(grep BUILD_TIME ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"

# collect tpu_mlir core files
export release_archive="./tpu_mlir"
rm -rf ${release_archive}
mkdir -p ${release_archive}
cp -rf ${INSTALL_PATH}/* ${release_archive}
rm ${release_archive}/python/mlir/_mlir_libs/libTPUMLIRPythonCAPI.so
# cp -rf ${PROJECT_ROOT}/regression ${release_archive}
# rm -rf ${release_archive}/regression/model
cp -rf ${PROJECT_ROOT}/third_party/customlayer ${release_archive}
mkdir -p ${release_archive}/lib/capi/
cp ${PROJECT_ROOT}/capi/lib/* ${release_archive}/lib/capi/

mkdir -p ${release_archive}/python/

cp -rf ${PROJECT_ROOT}/python/* ${release_archive}/python/
# batch convert import path -> absolute path
python3 ${PROJECT_ROOT}/release_tools/import_rewriter.py --project-root ${release_archive} --module-type tools
python3 ${PROJECT_ROOT}/release_tools/import_rewriter.py --project-root ${release_archive} --module-type utils

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

# remove docs
rm -rf ${release_archive}/docs
rm -rf ${release_archive}/ppl/doc

# remove redundant .so files
rm ${release_archive}/lib/libortools.so
rm ${release_archive}/lib/libortools.so.9.8.3296
rm ${release_archive}/lib/*.a

# collect_oneDNN_dependence
cp -L /usr/local/lib/libdnnl.so.3 ${release_archive}/lib/third_party/

# collect_capi_dependence
cp -rf ${PROJECT_ROOT}/capi/lib/* ${release_archive}/lib/third_party/

# automic entries gen for entry.py and set for setup.py
python ${release_archive}/entryconfig.py --execute_path bin/ python/tools/ python/samples/ python/test/ python/PerfAI/ --execute_file customlayer/test/test_custom_tpulang.py

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
    if [[ $file = *kernel_module*.so ]]; then
        echo Skip $file
        continue
    fi

    if [[ $file = *.a ]]; then
        echo Skip $file
        continue
    fi

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
