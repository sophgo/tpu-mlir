#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::Pow3Op::getFLOPs() { return module::getNumElements(getOutput()); }

LogicalResult top::Pow3Op::init(InferenceParameter &p) { return success(); }
void top::Pow3Op::deinit(InferenceParameter &p) {}

LogicalResult top::Pow3Op::inference(InferenceParameter &p) {
  auto num_element = module::getNumElements(getOutput());
#pragma omp parallel for schedule(static, omp_schedule(num_element))
  for (int i = 0; i < num_element; ++i) {
    auto val = p.inputs[0][i];
    auto ex = p.inputs[1][i];
    p.outputs[0][i] = std::pow(val, ex);
  }
  return success();
}

void top::Pow3Op::shape_inference() { common_shape_inference(getOperation()); }
