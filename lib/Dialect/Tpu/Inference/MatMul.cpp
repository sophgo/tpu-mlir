#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
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
  auto ldt = getDnnlType(input());
  auto rdt = getDnnlType(right());
  auto bdt = memory::data_type::f32;
  auto odt = memory::data_type::f32;
  if (with_bias) {
    bdt = getDnnlType(bias());
  }
  if (Quant::isUniformQuantized(output())) {
    odt = memory::data_type::s32;
  }

  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, relu, rshift(), multiplier(), ldt, rdt, bdt, odt);
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
  // llvm::errs() << "MatMulOp inference:" << this->name() << "\n";
  for (int i = 0; i < 5; i++) {
    // printf("%d  %f x %d +%f = %f\n", i, p.inputs[0][i],
    // (int8_t)p.inputs[1][i], p.inputs[2][i], p.outputs[0][i]);
  }
  return success();
}
