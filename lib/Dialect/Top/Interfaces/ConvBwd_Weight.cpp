#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::ConvBwdWeightOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::ConvBwdWeightOp::init(InferenceParameter &p) {
  return success();
}
void top::ConvBwdWeightOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConvBwdWeightOp::inference(InferenceParameter &p) {
  // UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::ConvBwdWeightOp::shape_inference() {}
