//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
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
        /*methodName=*/"codegen_global_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_bm168x",
        /*args=*/(ins),
        [{}],
        [{
          if (module::isBM1684Family()) {
            $_op.codegen_global_bm1684();
          } else if (module::isBM1684XFamily()
                    || module::isBM1690Family()) {
            $_op.codegen_global_bm1684x();
          }
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_global_cv18xx",
        /*args=*/(ins "int64_t":$layer_id)
      >,
      InterfaceMethod<
        /*desc=*/[{Check whether support multi core for global codegen}],
        /*retType=*/"bool",
        /*methodName=*/"support_multi_core",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_GLOBAL_CODEGEN_INTERFACE
