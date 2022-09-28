//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_TYPEINTERFACE
#define TPU_MLIR_TYPEINTERFACE

include "mlir/IR/OpBase.td"

def TypeInterface : OpInterface<"TypeInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          If Op's input type not the same with output type, then use this interface to do type transfer.
        }],
        /*retType=*/"mlir::Type",
        /*methodName=*/"type_verify",
        /*args=*/(ins "uint64_t":$opd_idx, "TypeCastMode&":$mode)
      >,
  ];
}
#endif // TPU_MLIR_TYPEINTERFACE
