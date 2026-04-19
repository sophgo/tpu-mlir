#!/usr/bin/env fish

set -l DIR (cd (dirname (status current-filename)); and pwd)

# it will be used in build.sh to remind people source envsetup.sh
# please keep it the same in both files
set -gx ENVSETUP_LAST_UPDATED "2025-05-22"

function __tpu_mlir_prepend_unique --argument-names var_name value
    if test -n "$value"
        if not contains -- $value $$var_name
            set -gx $var_name $value $$var_name
        end
    end
end

set -gx PROJECT_ROOT $DIR
if not set -q BUILD_PATH
    set -gx BUILD_PATH $PROJECT_ROOT/build
end
if not set -q INSTALL_PATH
    set -gx INSTALL_PATH $PROJECT_ROOT/install
end
set -gx TPUC_ROOT $INSTALL_PATH

echo "PROJECT_ROOT : $PROJECT_ROOT"
echo "BUILD_PATH   : $BUILD_PATH"
echo "INSTALL_PATH : $INSTALL_PATH"

# regression path
set -gx REGRESSION_PATH $PROJECT_ROOT/regression
set -gx NNMODELS_PATH $PROJECT_ROOT/../nnmodels
set -gx MODEL_ZOO_PATH $PROJECT_ROOT/../model-zoo

# customlayer path
set -gx CUSTOM_LAYER_PATH $PROJECT_ROOT/third_party/customlayer

# run path
__tpu_mlir_prepend_unique PATH $INSTALL_PATH/bin
__tpu_mlir_prepend_unique PATH $PROJECT_ROOT/python/tools
__tpu_mlir_prepend_unique PATH $PROJECT_ROOT/python/utils
__tpu_mlir_prepend_unique PATH $PROJECT_ROOT/python/test
__tpu_mlir_prepend_unique PATH $PROJECT_ROOT/python/samples
__tpu_mlir_prepend_unique PATH $PROJECT_ROOT/third_party/customlayer/python
__tpu_mlir_prepend_unique PATH $PROJECT_ROOT/python/tools/train/test

# ppl compiler path
set -gx PPL_PROJECT_ROOT $INSTALL_PATH/ppl
set -gx PPL_BUILD_PATH $PPL_PROJECT_ROOT/build
set -gx PPL_INSTALL_PATH $PPL_PROJECT_ROOT/install
set -gx PPL_RUNTIME_PATH $PPL_PROJECT_ROOT/deps
set -gx PPL_THIRD_PARTY_PATH $PPL_PROJECT_ROOT/third_party
__tpu_mlir_prepend_unique PATH $PPL_PROJECT_ROOT/bin
echo "PPL_PROJECT_ROOT : $PPL_PROJECT_ROOT"
echo "PPL_BUILD_PATH   : $PPL_BUILD_PATH"
echo "PPL_INSTALL_PATH : $PPL_INSTALL_PATH"

# other
set -gx CMODEL_LD_LIBRARY_PATH $INSTALL_PATH/lib $PROJECT_ROOT/capi/lib $LD_LIBRARY_PATH
set -gx CHIP_LD_LIBRARY_PATH /opt/sophon/libsophon-current/lib/ $INSTALL_PATH/lib $PROJECT_ROOT/capi/lib $LD_LIBRARY_PATH
set -gx LD_LIBRARY_PATH $CMODEL_LD_LIBRARY_PATH
set -gx USING_CMODEL True
__tpu_mlir_prepend_unique PYTHONPATH $INSTALL_PATH/python
__tpu_mlir_prepend_unique PYTHONPATH /usr/local/python_packages/
__tpu_mlir_prepend_unique PYTHONPATH $PROJECT_ROOT/python
__tpu_mlir_prepend_unique PYTHONPATH $PROJECT_ROOT/third_party/customlayer/python

set -gx OMP_NUM_THREADS 4
set -gx FORBID_GEN_RISCV_CODE 1

# CCache configuration
set -gx CCACHE_REMOTE_STORAGE redis://10.132.3.118:6379

# Coverage related settings
set -gx ENABLE_COVERAGE False

function enable_coverage --description "Enable code coverage for TPU-MLIR"
    set -gx ENABLE_COVERAGE True
    echo "Code coverage enabled"
end

function disable_coverage --description "Disable code coverage for TPU-MLIR"
    set -gx ENABLE_COVERAGE False
    echo "Code coverage disabled"
end

function use_cmodel --description "Use cmodel runtime libraries"
    set -gx USING_CMODEL True
    set -gx LD_LIBRARY_PATH $CMODEL_LD_LIBRARY_PATH
end

function use_chip --description "Use chip runtime libraries"
    set -gx USING_CMODEL False
    set -gx LD_LIBRARY_PATH $CHIP_LD_LIBRARY_PATH
end

function use_chip_cmodel --description "Mark the runtime as chip-backed cmodel"
    set -gx USING_CMODEL False
end

# only used to build libatomic_exec_aarch64.so for soc_infer
set -gx CROSS_TOOLCHAINS $PROJECT_ROOT/../bm_prebuilt_toolchains
set -gx LIBSOPHON_ROOT $PROJECT_ROOT/../libsophon

function rebuild_atomic_exec_alone --description "Rebuild libatomic_exec for soc_infer"
    set -gx ATOMIC_EXEC_ALONE 1
    if not set -q USE_CROSS_TOOLCHAINS
        if string match -q "*x86_64*" (uname -a)
            set -gx USE_CROSS_TOOLCHAINS 1
        end
    end

    set -l chiprunner_dir $PROJECT_ROOT/tools/chiprunner
    if not pushd $chiprunner_dir >/dev/null
        set -e ATOMIC_EXEC_ALONE
        return 1
    end

    rm -rf build
    mkdir build
    and cd build
    and cmake ..
    and make -j
    set -l build_status $status

    popd >/dev/null

    if test $build_status -ne 0
        set -e ATOMIC_EXEC_ALONE
        return $build_status
    end

    file $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_aarch64.so $PROJECT_ROOT/third_party/atomic_exec/libatomic_exec_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_aarch64.so $PROJECT_ROOT/install/lib/libatomic_exec_aarch64.so

    file $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_bm1688_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_bm1688_aarch64.so $PROJECT_ROOT/third_party/atomic_exec/libatomic_exec_bm1688_aarch64.so
    cp $PROJECT_ROOT/tools/chiprunner/build/libatomic_exec_bm1688_aarch64.so $PROJECT_ROOT/install/lib/libatomic_exec_bm1688_aarch64.so
    set -e ATOMIC_EXEC_ALONE
end

# insert hooks to .git/hooks
set -l SOURCE_DIR $PROJECT_ROOT/hooks
set -l TARGET_DIR $PROJECT_ROOT/.git/hooks
if test -d $TARGET_DIR
    echo "Install git hooks from $SOURCE_DIR to $TARGET_DIR"
    set -l hook_files (find $SOURCE_DIR -maxdepth 1 -type f -print)
    if test (count $hook_files) -gt 0
        cp -pu $hook_files $TARGET_DIR/
    end
end

functions -e __tpu_mlir_prepend_unique
