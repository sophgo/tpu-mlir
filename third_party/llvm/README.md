## clone and build llvm

``` shell
# 2022-06-04 13:00
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 15faac900d3b4fc17a04b3b9de421fab1bbe33db
mkdir -p llvm-project/build
git am 0001-ajust-for-tpu-mlir.patch
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_EH=ON \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
   -DCMAKE_BUILD_TYPE=Release \
   -DCMAKE_INSTALL_PREFIX=install
cmake --build . --target install

# use clean.sh to remove unused binary

