## clone and build llvm

``` shell
# 2023-04-22 15:00
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout c39dcf54bb0494f319cdc712d24607433bd7f30a
mkdir -p llvm-project/build
git am 0001-ajust-for-tpu-mlir.patch
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="" \
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

