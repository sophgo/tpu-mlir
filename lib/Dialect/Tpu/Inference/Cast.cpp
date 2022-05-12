#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::CastOp::init(InferenceParameter &p) { return success(); }
void tpu::CastOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CastOp::inference(InferenceParameter &p) {
  auto num_elem = Module::getNumElements(output());
  if (Quant::isUniformQuantized(output())) {
    auto qtype = Quant::getUniformQuantizedType(output());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      auto v = p.inputs[0][i] / qtype.getScale() + qtype.getZeroPoint();
      p.outputs[0][i] = Quant::to_int8(v);
    }
  } else if (Quant::isUniformQuantized(input())) {
    auto qtype = Quant::getUniformQuantizedType(input());
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
    for (size_t i = 0; i < num_elem; i++) {
      p.outputs[0][i] =
          qtype.getScale() * (p.inputs[0][i] - qtype.getZeroPoint());
    }
  }
  return success();
}
