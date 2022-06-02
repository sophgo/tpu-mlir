//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_INFERENCEINTERFACE
#define TPU_MLIR_INFERENCEINTERFACE

include "mlir/IR/OpBase.td"

def InferenceInterface : OpInterface<"InferenceInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"inference",
        /*args=*/(ins "InferenceParameter&":$param)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"init",
        /*args=*/(ins "InferenceParameter&":$param)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"deinit",
        /*args=*/(ins "InferenceParameter&":$param)
      >,
  ];
}
#endif // TPU_MLIR_INFERENCEINTERFACE
