#include "sophgo/Dialect/Top/IR/TopOps.h"
#include "sophgo/Interfaces/InferenceInterface.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

LogicalResult top::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N;
  bool with_bias, relu;
  parseParam(batch, M, K, N, with_bias, relu);
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, do_relu());
  p.handle = (void *)matmul;
  return success();
}

void top::MatMulOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto matmul = (MatMul *)p.handle;
    delete matmul;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::MatMulOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto matmul = (MatMul *)p.handle;
  matmul->run();
  return success();
}
