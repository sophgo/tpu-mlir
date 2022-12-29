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
#include "tpu_mlir/Support/Module.h"



// clang-format off
// case 1: [5, 6] * [6, 7] = [5, 7] => batch = 1, M = 5, K = 6, N = 7
// case 2: [1, 512, 7, 7] * [25088, 4096] = [1, 4096] => batch = 1, M = 1, K = 25088, N = 4096
// case 3: [3, 4, 5, 6] * [3, 4, 6, 7] = [3, 4, 5, 7] => batch = 12, M = 5, K = 6, N = 7
// case 4: [4, 5, 6] * [6,7] = [4, 5, 7] => batch =1, M = 20, K = 6, N = 7
// clang-format on
matmul_attr_t top::MatMulOp::parseParam() {
  matmul_attr_t p = {0};
  auto a_s = module::getShape(input());
  auto b_s = module::getShape(right());
  auto o_s = module::getShape(output());
  p.with_bias = !bias().getType().isa<mlir::NoneType>();
  p.do_relu = do_relu();
  p.relu_limit = this->relu_limit().convertToDouble();
  auto b_dims = b_s.size();
  auto o_dims = o_s.size();
  p.right_transpose = right_transpose();
  assert(b_dims >= 2);
  p.N = p.right_transpose ? b_s[b_dims - 2] : b_s[b_dims - 1];
  assert(p.N == o_s[o_dims - 1]);
  p.K = p.right_transpose ? b_s[b_dims - 1] : b_s[b_dims - 2];
  p.batch = 1;
  for (int i = 0; i < b_dims - 2; i++) {
    p.batch *= b_s[i];
  }
  if (p.batch > 1 || o_dims <= 2) {
    p.M = o_s[o_dims - 2];
  } else {
    p.M = std::accumulate(o_s.begin(), o_s.begin() + o_dims - 1, 1,
                          std::multiplies<int64_t>());
  }
  return p;
}

int64_t top::MatMulOp::getFLOPs() {
  auto p = parseParam();
  auto extra = p.with_bias ? 1 : 0 + p.do_relu ? 1 : 0;
  return p.batch * (2 * p.K + extra) * p.N * p.M;
}

LogicalResult top::MatMulOp::init(InferenceParameter &p) {
  auto matmul = new MatMul();
  auto a = parseParam();
  matmul->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.outputs[0], a.batch,
                a.M, a.K, a.N, a.do_relu, a.relu_limit, 0, a.right_transpose,
                0);
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
