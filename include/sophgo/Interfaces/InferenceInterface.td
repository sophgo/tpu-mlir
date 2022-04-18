#ifndef SOPHGO_INFERENCEINTERFACE
#define SOPHGO_INFERENCEINTERFACE

include "mlir/IR/OpBase.td"

def InferenceInterface : OpInterface<"InferenceInterface"> {
  let description = [{
  }];
  let cppNamespace = "::sophgo";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"inference",
        /*args=*/(ins "InferenceParameter&":$param)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"::mlir::LogicalResult",
        /*methodName=*/"init",
        /*args=*/(ins "InferenceParameter&":$param)
      >,
      InterfaceMethod<
        /*desc=*/[{
        }],
        /*retType=*/"void",
        /*methodName=*/"deinit",
        /*args=*/(ins "InferenceParameter&":$param)
      >,
  ];
}
#endif // SOPHGO_INFERENCEINTERFACE
