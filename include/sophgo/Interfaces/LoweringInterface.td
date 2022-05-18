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
        /*methodName=*/"lowering_fp32_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_fp32_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"lowering_bf16_bm1686",
        /*args=*/(ins),
        [{}],
        [{
          llvm_unreachable("not support now");
          return nullptr;
        }]
      >,
  ];

}
#endif // SOPHGO_LOWERINGINTERFACE
