#ifndef SOPHGO_FLOPSINTERFACE
#define SOPHGO_FLOPSINTERFACE

include "mlir/IR/OpBase.td"

def FlopsInterface : OpInterface<"FlopsInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"int64_t",
        /*methodName=*/"getFLOPs",
        /*args=*/(ins)
      >,
  ];
}
#endif // SOPHGO_FLOPSINTERFACE
