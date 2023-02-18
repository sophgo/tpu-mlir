//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_ELTWISEINTERFACE
#define TPU_MLIR_ELTWISEINTERFACE

include "mlir/IR/OpBase.td"

def EltwiseInterface : OpInterface<"EltwiseInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"bool",
        /*methodName=*/"isEltwise",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_ELTWISEINTERFACE
