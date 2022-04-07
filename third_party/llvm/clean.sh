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
  "libLLVMBinaryFormat.a"
  "libLLVMRemarks.a"
  "libLLVMSupport.a"
  "libLLVMDemangle.a"
  "libLLVMBitstreamReader.a"
  "libMLIRAnalysis.a"
  "libMLIRArithmetic.a"
  "libMLIRCallInterfaces.a"
  "libMLIRControlFlow.a"
  "libMLIRCopyOpInterface.a"
  "libMLIRDataLayoutInterfaces.a"
  "libMLIRViewLikeInterface.a"
  "libMLIRFunc.a"
  "libMLIRIR.a"
  "libMLIRControlFlowInterfaces.a"
  "libMLIRDialect.a"
  "libMLIRLoopLikeInterface.a"
  "libMLIRInferTypeOpInterface.a"
  "libMLIROptLib.a"
  "libMLIRQuant.a"
  "libMLIRQuantTransforms.a"
  "libMLIRQuantUtils.a"
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

