//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#ifndef SOPHGO_WEIGHTREORDERINTERFACE
#define SOPHGO_WEIGHTREORDERINTERFACE

include "mlir/IR/OpBase.td"

def WeightReorderInterface : OpInterface<"WeightReorderInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
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
        /*methodName=*/"weight_reorder_float_bm1686",
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
          if (chip == Module::Chip::BM1684) {
            if (Module::isFloatType($_op.getResult())) {
            } else {
              $_op.weight_reorder_int8_bm1684();
            }
          } else if (chip == Module::Chip::BM1686) {
            if (Module::isFloatType($_op.getResult())) {
              $_op.weight_reorder_float_bm1686();
            } else {
              $_op.weight_reorder_int8_bm1686();
            }
          }
        }]
      >,
  ];
}
#endif // SOPHGO_WEIGHTREORDERINTERFACE
