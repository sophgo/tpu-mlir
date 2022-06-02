//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_GLOBAL_CODEGEN_INTERFACE
#define TPU_MLIR_GLOBAL_CODEGEN_INTERFACE

include "mlir/IR/OpBase.td"

def GlobalGenInterface : OpInterface<"GlobalGenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_int8_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_float_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global",
        /*args=*/(ins),
        [{}],
        [{
          auto module = Module::getModuleOp($_op);
          auto chip = Module::getChip(module);
          if (chip == Module::Chip::BM1684) {
            $_op.codegen_global_int8_bm1684();
          } else if (chip == Module::Chip::BM1686) {
            if (Module::isFloatType($_op.getResult())) {
              $_op.codegen_global_float_bm1686();
            } else {
              $_op.codegen_global_int8_bm1686();
            }
          }
        }]
      >,
  ];
}
#endif // TPU_MLIR_GLOBAL_CODEGEN_INTERFACE
