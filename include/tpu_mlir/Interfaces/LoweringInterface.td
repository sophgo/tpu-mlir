//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_LOWERINGINTERFACE
#define TPU_MLIR_LOWERINGINTERFACE

include "mlir/IR/OpBase.td"

def LoweringInterface : OpInterface<"LoweringInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_f32_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_int8_bm1684x",
        /*args=*/(ins "bool":$asymmetric)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_quant_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_f32_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_f16_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_bf16_bm1684x",
        /*args=*/(ins)
      >,
      ];

}
#endif // TPU_MLIR_LOWERINGINTERFACE
