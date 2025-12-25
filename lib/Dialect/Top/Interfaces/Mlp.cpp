#include "tpu_mlir/Support/Dnnl/Dnnl.h"

mlp_attr_t top::MlpOp::parseParam() {
  mlp_attr_t p = {0};
  return p;
}

int64_t top::MlpOp::getFLOPs() { return 0; }

LogicalResult top::MlpOp::init(InferenceParameter &p) { return success(); }

void top::MlpOp::deinit(InferenceParameter &p) { return; }

LogicalResult top::MlpOp::inference(InferenceParameter &p) { return success(); }

void top::MlpOp::shape_inference() { common_shape_inference(getOperation()); }
