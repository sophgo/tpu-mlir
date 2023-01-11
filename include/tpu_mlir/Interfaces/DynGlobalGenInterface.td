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
        /*retType=*/"int64_t",
        /*methodName=*/"dyn_codegen_global_bm1684",
        /*args=*/(ins "void *":$buffer),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return 0;
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"dyn_codegen_global_bm1684x",
        /*args=*/(ins "void *":$buffer)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"int64_t",
        /*methodName=*/"dyn_codegen_global_cv18xx",
        /*args=*/(ins "void *":$buffer),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return 0;
        }]
      >,
  ];
}
#endif // TPU_MLIR_DYN_GLOBAL_CODEGEN_INTERFACE
