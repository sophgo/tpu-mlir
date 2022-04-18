#ifndef SOPHGO_QUANTIZEINTERFACE
#define SOPHGO_QUANTIZEINTERFACE

include "mlir/IR/OpBase.td"

def QuantizeInterface : OpInterface<"QuantizeInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"quantize_int8_bm1684",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"quantize_int8_bm1686",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"quantize_bf16_bm1686",
        /*args=*/(ins),
        [{}],
        [{
          llvm_unreachable("not support now");
          return nullptr;
        }]
      >,
  ];

}
#endif // SOPHGO_QUANTIZEINTERFACE
