#!/bin/bash
set -xe
shopt -s extglob
pushd bin/
for file in `ls !(mlir-tblgen)`; do echo "" > $file; done
popd

mkdir need
pushd lib/
need_list=(
  "objects*"
  "cmake"
  "libLLVMCore.a"
  "libLLVMSupport.a"
  "libMLIRAnalysis.a"
  "libMLIRArithmetic.a"
  "libMLIRCallInterfaces.a"
  "libMLIRControlFlow.a"
  "libMLIRFunc.a"
  "libMLIRIR.a"
  "libMLIRControlFlowInterfaces.a"
  "libMLIRDialect.a"
  "libMLIRInferTypeOpInterface.a"
  "libMLIROptLib.a"
  "libMLIRParser.a"
  "libMLIRPass.a"
  "libMLIRPDL.a"
  "libMLIRPDLInterp.a"
  "libMLIRPDLToPDLInterp.a"
  "libMLIRRewrite.a"
  "libMLIRSideEffectInterfaces.a"
  "libMLIRSupport.a"
  "libMLIRTransforms.a"
  "libMLIRTransformUtils.a"
)

for file in ${need_list[@]}
do
  mv $file ../need/
done
for file in `ls`; do echo "" > $file; done
mv ../need/* .
popd
rm -rf need

