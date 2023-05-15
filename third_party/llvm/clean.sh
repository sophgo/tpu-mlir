#!/bin/bash
set -xe
pushd bin/
for file in `ls | grep -vw -E 'mlir-tblgen|mlir-translate|llc|mlir-opt'`; do echo "" > $file; done
popd

pushd include/mlir/Dialect/
for file in `ls | grep -vw -E 'Func|Quant|IRDL|PDL|PDLInterp|Tosa|CommonFolders.h|Traits.h'`; do rm $file -rf; done
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
  "libMLIRIRDL.a"
  "libMLIRTosaDialect.a"
  "libMLIRPluginsLib.a"
  "libMLIRCastInterfaces.a"
  "libMLIRDialectUtils.a"
  "libMLIRInferIntRangeCommon.a"
  "libMLIRPresburger.a"
  "libMLIRLoopLikeInterface.a"
  "libMLIRInferTypeOpInterface.a"
  "libMLIROptLib.a"
  "libMLIRDebug.a"
  "libMLIRQuantDialect.a"
  "libMLIRQuantUtils.a"
  "libMLIRMemorySlotInterfaces.a"
  "libMLIRParser.a"
  "libMLIRPass.a"
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

