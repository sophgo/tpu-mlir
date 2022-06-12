//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_WEIGHTREORDERINTERFACE
#define TPU_MLIR_WEIGHTREORDERINTERFACE

include "mlir/IR/OpBase.td"

def WeightReorderInterface : OpInterface<"WeightReorderInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_int8_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_bf16_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_f16_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_f32_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder",
        /*args=*/(ins),
        [{}],
        [{
          auto module = Module::getModuleOp($_op);
          auto chip = Module::getChip(module);
          auto type = Module::getStorageType($_op.getResult());
          if (chip == Module::Chip::BM1684) {
            $_op.weight_reorder_int8_bm1684();
          } else if (chip == Module::Chip::BM1686) {
            if (type.isF32()) {
              $_op.weight_reorder_f32_bm1686();
            } else if (type.isF16()) {
              $_op.weight_reorder_f16_bm1686();
            } else if (type.isBF16()) {
              $_op.weight_reorder_bf16_bm1686();
            } else {
              $_op.weight_reorder_int8_bm1686();
            }
          }
        }]
      >,
  ];
}
#endif // TPU_MLIR_WEIGHTREORDERINTERFACE
