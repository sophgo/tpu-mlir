#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::VarianceOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::VarianceOp::init(InferenceParameter &p) { return success(); }
void top::VarianceOp::deinit(InferenceParameter &p) {}

LogicalResult top::VarianceOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::VarianceOp::shape_inference() {}
