#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult tpu::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N;
  bool relu, with_bias;
  parseParam(batch, M, K, N, with_bias, relu);

  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, relu);
  p.handle = (void *)matmul;
  return success();
}

void tpu::MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;
  matmul->run();
  if (Quant::isUniformQuantized(output())) {
    auto o_qtype = Quant::getUniformQuantizedType(output());
    auto rft = rshift();
    auto mlti = multiplier();
    auto num_output = Module::getNumElements(output());
  #pragma omp parallel for schedule(static, omp_schedule(num_output))
    for (int64_t i = 0; i < num_output; i++) {
      auto v = (((int64_t)(p.outputs[0][i] * mlti)) >> rft);
      p.outputs[0][i] = Quant::to_int8(v + o_qtype.getZeroPoint());
    }
  }

  return success();
}
