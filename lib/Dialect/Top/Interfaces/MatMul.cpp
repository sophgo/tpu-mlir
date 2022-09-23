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

void top::MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
                               int64_t &N, bool &with_bias, bool &relu,
                               double &limit) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto r_s = right().getType().cast<RankedTensorType>().getShape();
  with_bias = !bias().getType().isa<mlir::NoneType>();
  relu = do_relu();
  limit = this->relu_limit().convertToDouble();
  auto r_dims = r_s.size();
  auto i_dims = i_s.size();
  N = r_s[r_dims - 1];
  K = r_s[r_dims - 2];
  if (r_dims > 2) {
    M = i_s[i_dims - 2];
    assert(i_s[i_dims - 1] == K);
    batch = std::accumulate(r_s.begin(), r_s.begin() + r_dims - 2, 1,
                            std::multiplies<int64_t>());
  } else {
    batch = 1;
    M = std::accumulate(i_s.begin(), i_s.begin() + i_dims - 1, 1,
                        std::multiplies<int64_t>());
  }
}

int64_t top::MatMulOp::getFLOPs() {
  int64_t batch, M, K, N;
  bool has_relu, with_bias;
  double limit;
  parseParam(batch, M, K, N, with_bias, has_relu, limit);
  auto extra = with_bias ? 1 : 0 + has_relu ? 1 : 0;
  return batch * (2 * K + extra) * N * M;
}

LogicalResult top::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  int64_t batch, M, K, N;
  bool with_bias, relu;
  double limit;
  parseParam(batch, M, K, N, with_bias, relu, limit);
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], batch, M,
                K, N, relu, limit, 0);
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
