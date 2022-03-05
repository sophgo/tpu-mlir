## clone and build llvm

``` shell
# 2022-03-05 12:30
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout f909aed671fee194297ba9190f0e4baa4b10d0c2 && cd ..
mkdir -p llvm-project/build
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

