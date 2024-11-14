#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::LogicalAndOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::LogicalAndOp::init(InferenceParameter &p) {
  return success();
}
void top::LogicalAndOp::deinit(InferenceParameter &p) {}

LogicalResult top::LogicalAndOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
  return success();
}

void top::LogicalAndOp::shape_inference() {}
