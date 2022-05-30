#ifndef SOPHGO_LOWERINGINTERFACE
#define SOPHGO_LOWERINGINTERFACE

include "mlir/IR/OpBase.td"

def LoweringInterface : OpInterface<"LoweringInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_int8_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_fp",
        /*args=*/(ins "llvm::StringRef":$mode)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_f32_bm1684",
        /*args=*/(ins)
      >  ];

}
#endif // SOPHGO_LOWERINGINTERFACE
