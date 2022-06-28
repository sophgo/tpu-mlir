//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TRAITS
#define TPU_MLIR_TRAITS

include "mlir/IR/OpBase.td"

class TPUMLIROpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "::tpu_mlir::trait";
}

def SupportFuseRelu : TPUMLIROpTrait<"SupportFuseRelu">;
def HasCommonAttributes : TPUMLIROpTrait<"HasCommonAttributes">;
def TpuTypeRestrict: TPUMLIROpTrait<"TpuTypeRestrict">;
def InOutSameShape: TPUMLIROpTrait<"InOutSameShape">;
#endif
