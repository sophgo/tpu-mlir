//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_OPS_INTERFACES
#define STRUCTURED_OPS_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def StructuredInterface : OpInterface<"StructuredOp"> {
  let cppNamespace = "::tpu_mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Return true if all the indexing maps are projected permutations.
        Otherwise return false.
      }],
      /*retTy=*/"mlir::LogicalResult",
      /*methodName=*/"verifyAndCodeGen"
    >
  ];
}

#endif // STRUCTURED_OPS_INTERFACES
