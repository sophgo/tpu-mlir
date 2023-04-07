//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#ifndef TPU_MLIR_DYN_GLOBAL_CODEGEN_INTERFACE
#define TPU_MLIR_DYN_GLOBAL_CODEGEN_INTERFACE

include "mlir/IR/OpBase.td"
def DynGlobalGenInterface : OpInterface<"DynGlobalGenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::tpu_mlir";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"uint32_t",
        /*methodName=*/"dyn_codegen_global_bm1684",
        /*args=*/(ins "void *":$ir_layer_info)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"dyn_codegen_global_bm1684x",
        /*args=*/(ins "void *":$buffer)
      >,
      InterfaceMethod<
        /*desc=*/"Return the layer type.",
        /*retType=*/"int64_t",
        /*methodName=*/"get_fw_type_bm1684x",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/"Return the layer type.",
        /*retType=*/"int64_t",
        /*methodName=*/"get_fw_type_bm1684",
        /*args=*/(ins)
      >,
  ];
}
#endif // TPU_MLIR_DYN_GLOBAL_CODEGEN_INTERFACE
