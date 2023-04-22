#!/bin/bash
set -xe
shopt -s extglob
pushd bin/
for file in `ls !(mlir-tblgen)`; do echo "" > $file; done
popd

rm -rf need
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
  "libLLVMTargetParser.a"
  "libMLIRBytecodeReader.a"
  "libMLIRBytecodeWriter.a"
  "libMLIRAsmParser.a"
  "libMLIRPDLDialect.a"
  "libMLIRPDLInterpDialect.a"
  "libMLIRAnalysis.a"
  "libMLIRCallInterfaces.a"
  "libMLIRCopyOpInterface.a"
  "libMLIRDataLayoutInterfaces.a"
  "libMLIRControlFlowInterfaces.a"
  "libMLIRRuntimeVerifiableOpInterface.a"
  "libMLIRViewLikeInterface.a"
  "libMLIRFuncDialect.a"
  "libMLIRIR.a"
  "libMLIRDialect.a"
  "libMLIRControlFlowDialect.a"
  "libMLIRTosaDialect.a"
  "libMLIRAffineDialect.a"
  "libMLIRSCFDialect.a"
  "libMLIRTensorDialect.a"
  "libMLIRArithUtils.a"
  "libMLIRParallelCombiningOpInterface.a"
  "libMLIRPluginsLib.a"
  "libMLIRLinalgDialect.a"
  "libMLIRDestinationStyleOpInterface.a"
  "libMLIRValueBoundsOpInterface.a"
  "libMLIRComplexDialect.a"
  "libMLIRCastInterfaces.a"
  "libMLIRDialectUtils.a"
  "libMLIRArithDialect.a"
  "libMLIRInferIntRangeCommon.a"
  "libMLIRPresburger.a"
  "libMLIRMemRefDialect.a"
  "libMLIRLoopLikeInterface.a"
  "libMLIRInferTypeOpInterface.a"
  "libMLIROptLib.a"
  "libMLIRDebug.a"
  "libMLIRQuantDialect.a"
  "libMLIRQuantUtils.a"
  "libMLIRParser.a"
  "libMLIRPass.a"
  "libMLIRPDL.a"
  "libMLIRPDLInterp.a"
  "libMLIRPDLToPDLInterp.a"
  "libMLIRObservers.a"
  "libMLIRRewrite.a"
  "libMLIRSideEffectInterfaces.a"
  "libMLIRInferIntRangeInterface.a"
  "libMLIRSupport.a"
  "libMLIRTransforms.a"
  "libMLIRTransformUtils.a"
)

for file in ${need_list[@]}
do
  if [ -e $file ]; then
    mv $file ../need/
  else
    echo "[Error] $file not exist"
  fi
done
for file in `ls`; do echo "" > $file; done
mv ../need/* .
popd
rm -rf need

