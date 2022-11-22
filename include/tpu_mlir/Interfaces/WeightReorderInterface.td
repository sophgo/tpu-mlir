//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
        /*methodName=*/"weight_reorder_int8_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_int8_cv18xx",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_bf16_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_bf16_cv18xx",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_f16_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"weight_reorder_f32_bm1684x",
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
          auto type = Module::getStorageType($_op->getResult(0));
          if (Module::isBM1684Family(chip)) {
            $_op.weight_reorder_int8_bm1684();
          } else if (Module::isBM1684XFamily(chip)) {
            if (type.isF32()) {
              $_op.weight_reorder_f32_bm1684x();
            } else if (type.isF16()) {
              $_op.weight_reorder_f16_bm1684x();
            } else if (type.isBF16()) {
              $_op.weight_reorder_bf16_bm1684x();
            } else {
              $_op.weight_reorder_int8_bm1684x();
            }
          } else if (chip == Module::Chip::CV182x || chip == Module::Chip::CV183x) {
            if (type.isBF16()) {
              $_op.weight_reorder_bf16_cv18xx();
            } else {
              $_op.weight_reorder_int8_cv18xx();
            }
          }
        }]
      >,
  ];
}
#endif // TPU_MLIR_WEIGHTREORDERINTERFACE
