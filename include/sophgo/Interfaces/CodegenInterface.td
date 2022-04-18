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
        /*methodName=*/"codegen_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{}],
        /*retType=*/"void",
        /*methodName=*/"codegen_int8_bm1686",
        /*args=*/(ins)
      >,
  ];
}
#endif // SOPHGO_CODEGENINTERFACE
