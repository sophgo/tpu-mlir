#ifndef GLOBAL_OPTIMIZATION_PASSES
#define GLOBAL_OPTIMIZATION_PASSES

include "mlir/Pass/PassBase.td"
def DetachElementwiseFromNamedOps :
    Pass<"detach-elementwise-from-named-ops", ""> {
  let summary = "Detaches elementwise ops from named Linalg ops";
  let constructor = "mlir::createDetachElementwiseFromNamedOpsPass()";
}

def RemoveZeroExtentTensors :
    InterfacePass<"remove-zero-extent-tensors", "mlir::FunctionOpInterface"> {
  let summary = "Remove tensors that have 0-extents";
  let constructor = "mlir::createRemoveZeroExtentTensorsPass()";
}

def Convert1X1FilterConv2DToMatmul:
    Pass<"convert-1x1-filter-conv2d-to-matmul", ""> {
  let summary = "Convert linalg convolution ops with 1x1 kernels into linalg matrix multiplication ops.";
  let constructor = "mlir::createConvert1X1FilterConv2DToMatmulPass()";
}

def EraseUnusedLinalgOperands :
    Pass<"erase-unused-linalg-operands", "mlir::ModuleOp"> {
  let summary = "Erase unused linalg operand and remove dead code.";
  let constructor = "mlir::createEraseUnusedLinalgOperands()";
}

def GeneralizeLinalgNamedOps :
    InterfacePass<"generalize-linalg-named-ops", "mlir::FunctionOpInterface"> {
  let summary = "Convert some Linalg named ops into linalg.generics";
  let constructor = "mlir::createGeneralizeLinalgNamedOpsPass()";
}
#endif
