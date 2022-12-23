//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// clang-format on
void top::MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
                               int64_t &N, bool &with_bias, bool &relu,
                               double &limit, bool &transpose) {
  auto a_s = Module::getShape(input());
  auto b_s = Module::getShape(right());
  auto o_s = Module::getShape(output());
  with_bias = !bias().getType().isa<mlir::NoneType>();
  relu = do_relu();
  limit = this->relu_limit().convertToDouble();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  transpose = right_transpose();
  assert(b_dims >= 2);
  N = transpose ? b_s[b_dims - 2] : b_s[b_dims - 1];
  assert(N == o_s[o_dims - 1]);
  K = transpose ? b_s[b_dims - 1] : b_s[b_dims - 2];
  batch = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    batch *= b_s[i];
  }
  if (batch > 1 || o_dims <= 2) {
    M = o_s[o_dims - 2];
  } else {
    M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                        std::multiplies<int64_t>());
  }
}

int64_t top::MatMulOp::getFLOPs() {
  int64_t batch, M, K, N;
  bool has_relu, with_bias, transpose;
  double limit;
  parseParam(batch, M, K, N, with_bias, has_relu, limit, transpose);
  auto extra = with_bias ? 1 : 0 + has_relu ? 1 : 0;
  return batch * (2 * K + extra) * N * M;
}

LogicalResult top::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N;
  bool with_bias, relu, right_transpose;
  double limit;
  parseParam(batch, M, K, N, with_bias, relu, limit, right_transpose);
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, relu, limit, 0, right_transpose, 0);
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
