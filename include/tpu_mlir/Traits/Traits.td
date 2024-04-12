//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TRAITS
#define TPU_MLIR_TRAITS

include "mlir/IR/OpBase.td"

class TPUMLIROpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::tpu_mlir::trait";
}

def ShapeProducer : TPUMLIROpTrait<"ShapeProducer">;
def ShapeConsumer : TPUMLIROpTrait<"ShapeConsumer">;
def ScalarProducer : TPUMLIROpTrait<"ScalarProducer">;
def ScalarConsumer : TPUMLIROpTrait<"ScalarConsumer">;
def SupportPermuteMove : TPUMLIROpTrait<"SupportPermuteMove">;
def SupportConstant: TPUMLIROpTrait<"SupportConstant">;
def SupportFuseRelu : TPUMLIROpTrait<"SupportFuseRelu">;
def SupportEarlyStride : TPUMLIROpTrait<"SupportEarlyStride">;
def SupportElementwise : TPUMLIROpTrait<"SupportElementwise">;
def TpuTypeRestrict: TPUMLIROpTrait<"TpuTypeRestrict">;
def InOutSameShape: TPUMLIROpTrait<"InOutSameShape">;

#endif
