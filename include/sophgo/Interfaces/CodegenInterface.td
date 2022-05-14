#ifndef SOPHGO_CODEGENINTERFACE
#define SOPHGO_CODEGENINTERFACE

include "mlir/IR/OpBase.td"

def CodegenInterface : OpInterface<"CodegenInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
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
        /*methodName=*/"codegen_local_int8_bm1684",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_local_int8_bm1686",
        /*args=*/(ins "int64_t":$n_step, "int64_t":$h_step)
      >,
  ];
}
#endif // SOPHGO_CODEGENINTERFACE
