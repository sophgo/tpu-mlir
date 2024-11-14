#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::RsqrtOp::getFLOPs() {
  return module::getNumElements(getOutput()) * 2;
}

LogicalResult top::RsqrtOp::init(InferenceParameter &p) { return success(); }
void top::RsqrtOp::deinit(InferenceParameter &p) {}

LogicalResult top::RsqrtOp::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
  float eps = 1e-5;
  float molecular = 1.0;
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    auto sqrt = std::sqrt(val + eps);
    p.outputs[0][i] = molecular / sqrt;
  }
  return success();
}

void top::RsqrtOp::shape_inference() { common_shape_inference(getOperation()); }
