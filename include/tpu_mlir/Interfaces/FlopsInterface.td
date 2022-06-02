//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_FLOPSINTERFACE
#define TPU_MLIR_FLOPSINTERFACE

include "mlir/IR/OpBase.td"

def FlopsInterface : OpInterface<"FlopsInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getFLOPs",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_FLOPSINTERFACE
