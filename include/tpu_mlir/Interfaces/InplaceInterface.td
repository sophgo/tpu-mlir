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
        /*desc=*/[{ get the result index according to the operand index,
                     if the result is not inplace, return -1 }],
        /*retType=*/"int64_t",
        /*methodName=*/"get_inplace_result_index",
        /*args=*/(ins "int64_t":$opd_index),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return -1;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{ get the operand index according to the result index,
                     if the operand is not inplace, return -1 }],
        /*retType=*/"int64_t",
        /*methodName=*/"get_inplace_operand_index",
        /*args=*/(ins "int64_t":$result_index),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return -1;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"bool",
        /*methodName=*/"SupportInplace",
        /*args=*/(ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return false;
        }]
      >,
  ];
}
#endif // TPU_MLIR_INPLACE_INTERFACE
