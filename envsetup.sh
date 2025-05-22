#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

# it will be used in build.sh to remind people source envsetup.sh
# please keep it the same in both files
export ENVSETUP_LAST_UPDATED="2025-05-22"

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

# customlayer path
export CUSTOM_LAYER_PATH=$PROJECT_ROOT/third_party/customlayer

# run path
export PATH=$INSTALL_PATH/bin:$PATH
export PATH=$PROJECT_ROOT/python/tools:$PATH
export PATH=$PROJECT_ROOT/python/utils:$PATH
export PATH=$PROJECT_ROOT/python/test:$PATH
export PATH=$PROJECT_ROOT/python/samples:$PATH
export PATH=$PROJECT_ROOT/third_party/customlayer/python:$PATH
export PATH=$PROJECT_ROOT/python/tools/train/test:$PATH

# ppl compiler path
export PPL_PROJECT_ROOT=$INSTALL_PATH/ppl
export PPL_BUILD_PATH=$PPL_PROJECT_ROOT/build
export PPL_INSTALL_PATH=$PPL_PROJECT_ROOT/install
export PPL_RUNTIME_PATH=$PPL_PROJECT_ROOT/runtime
export PPL_THIRD_PARTY_PATH=$PPL_PROJECT_ROOT/third_party
export PATH=$PPL_PROJECT_ROOT/bin:$PATH
echo "PPL_PROJECT_ROOT : ${PPL_PROJECT_ROOT}"
echo "PPL_BUILD_PATH   : ${PPL_BUILD_PATH}"
echo "PPL_INSTALL_PATH : ${PPL_INSTALL_PATH}"

# other
export CMODEL_LD_LIBRARY_PATH=$INSTALL_PATH/lib:$PROJECT_ROOT/capi/lib:$LD_LIBRARY_PATH
export CHIP_LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib/:$INSTALL_PATH/lib:$PROJECT_ROOT/capi/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CMODEL_LD_LIBRARY_PATH
export USING_CMODEL=True
export PYTHONPATH=$INSTALL_PATH/python:$PYTHONPATH
export PYTHONPATH=/usr/local/python_packages/:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/python:$PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT/third_party/customlayer/python:$PYTHONPATH


export OMP_NUM_THREADS=4

export FORBID_GEN_RISCV_CODE=1

# CCache configuration
export CCACHE_REMOTE_STORAGE=redis://10.132.3.118:6379

# 覆盖率相关设置
export ENABLE_COVERAGE=False
function enable_coverage() {
    export ENABLE_COVERAGE=True
    echo "Code coverage enabled"
}
function disable_coverage() {
    export ENABLE_COVERAGE=False
    echo "Code coverage disabled"
}

function use_cmodel() {
    export USING_CMODEL=True
    export LD_LIBRARY_PATH=$CMODEL_LD_LIBRARY_PATH
}
function use_chip() {
    export USING_CMODEL=False
    export LD_LIBRARY_PATH=$CHIP_LD_LIBRARY_PATH
}
function use_chip_cmodel() {
    export USING_CMODEL=False
}
function use_8ch() {
    export CMODEL_GLOBAL_MEM_SIZE=137438953472
    export USING_8CH=1
    sysctl vm.overcommit_memory=1
}
function use_32ch() {
    unset CMODEL_GLOBAL_MEM_SIZE
    unset USING_8CH
    sysctl vm.overcommit_memory=0
}

# only used to build libatomic_exec_aarch64.so for soc_infer
export CROSS_TOOLCHAINS=$PROJECT_ROOT/../bm_prebuilt_toolchains
export LIBSOPHON_ROOT=$PROJECT_ROOT/../libsophon
function rebuild_atomic_exec_alone() {
    export ATOMIC_EXEC_ALONE=1
    if [ -z "$USE_CROSS_TOOLCHAINS" ]; then
        if [ $(uname -a | grep -q "x86_64") ]; then
            export USE_CROSS_TOOLCHAINS=1
        fi
    fi

    pushd $PROJECT_ROOT/tools/chiprunner/
    rm -rf build
    mkdir build && cd build
    cmake ..
    make -j
    popd
    file $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_aarch64.so $PROJECT_ROOT/third_party/atomic_exec/libatomic_exec_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_aarch64.so $PROJECT_ROOT/install/lib/libatomic_exec_aarch64.so

    file $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_bm1688_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_bm1688_aarch64.so $PROJECT_ROOT/third_party/atomic_exec/libatomic_exec_bm1688_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_bm1688_aarch64.so $PROJECT_ROOT/install/lib/libatomic_exec_bm1688_aarch64.so
    unset ATOMIC_EXEC_ALONE
}

# insert hooks to .git/hooks
SOURCE_DIR="hooks"
TARGET_DIR=".git/hooks"
if [ -d "$TARGET_DIR" ]; then
    echo "Install git hooks from $SOURCE_DIR to $TARGET_DIR"
    cp -pu "$SOURCE_DIR/"* "$TARGET_DIR/"
fi
