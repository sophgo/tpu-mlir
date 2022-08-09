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
        /*methodName=*/"codegen_global_int8_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_float_bm1684x",
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
          auto type = Module::getStorageType($_op.getResult());
          if (chip == Module::Chip::BM1684) {
            $_op.codegen_global_int8_bm1684();
          } else if (chip == Module::Chip::BM1684x) {
            if (type.isF32() || type.isF16() || type.isBF16()) {
              $_op.codegen_global_float_bm1684x();
            } else {
              $_op.codegen_global_int8_bm1684x();
            }
          }
        }]
      >,
  ];
}
#endif // TPU_MLIR_GLOBAL_CODEGEN_INTERFACE
