//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_INPLACE_INTERFACE
#define TPU_MLIR_INPLACE_INTERFACE

include "mlir/IR/OpBase.td"

def InplaceInterface : OpInterface<"InplaceInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"bool",
        /*methodName=*/"supportInplace",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_INPLACE_INTERFACE
