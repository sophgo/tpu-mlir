## clone and build llvm

``` shell
# 2022-10-04 19:00
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 2e87333bfe7e226533fb66c84015c4523e352d3f
mkdir -p llvm-project/build
git am 0001-ajust-for-tpu-mlir.patch
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_EH=ON \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_INCLUDE_TESTS=OFF \
   -DMLIR_INCLUDE_TESTS=OFF \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=llvm
cmake --build . --target install

# use clean.sh to remove unused binary

