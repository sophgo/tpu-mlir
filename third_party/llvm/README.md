## clone and build llvm

``` shell
# 2022-04-07 11:00
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout f2796a5d444998ea73f02f433ed34b7c09e0f7d5 && cd ..
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

