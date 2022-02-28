## clone and build llvm

``` shell
# 2022-02-21 10:30
git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout efe5b8ad904bfb1d9abe6ac7123494b534040238 && cd ..
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

# remove all unused bin, only keep mlir-tblgen
shopt -s extglob
pushd install/bin/
for file in `ls !(mlir-tblgen)`; do echo "" > $file; done
popd
# remove all unused lib
# keep:
# libLLVMCore.a libLLVMSupport.a obj* cmake libMLIRAnalysis.a
# libMLIRArithmetic.a libMLIRCallInterfaces.a libMLIRControlFlow.a libMLIRIR.a
# libMLIRControlFlowInterfaces.a libMLIRDialect.a libMLIRInferTypeOpInterface.a
# libMLIRMlirOptMain.a libMLIROptLib.a libMLIRParser.a libMLIRPass.a libMLIRPDL.a
# libMLIRPDLInterp.a libMLIRPDLLParser.a libMLIRPDLToPDLInterp.a libMLIRRewrite.a
# libMLIRSideEffectInterfaces.a libMLIRStandard.a libMLIRSupport.a
# libMLIRTransforms.a libMLIRTransformUtils.a
pushd install/lib/
for file in `ls !(libMLIR*.a|libLLVMCore.a|libLLVMSupport.a|obj*|cmake|...)`; do echo "" > $file; done

popd
```
