//===----------------------------------------------------------------------===//
//
// Copyright (c) 2020-2030 by Sophgo Technologies Inc. All rights reserved.
//
// Licensed under the Apache License v2.0.
// See http://www.apache.org/licenses/LICENSE-2.0 for license information.
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//

#include "sophgo/Dialect/Tpu/IR/TpuOps.h"
#include "sophgo/Support/Dnnl/Dnnl.h"
#include "sophgo/Support/Helper/Quant.h"
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace sophgo::helper;
using namespace mlir;

void tpu::MatMulOp::parseParam(int64_t &batch, int64_t &M, int64_t &K,
                               int64_t &N, bool &with_bias, bool &relu) {
  auto i_s = input().getType().cast<RankedTensorType>().getShape();
  auto r_s = right().getType().cast<RankedTensorType>().getShape();
  auto o_s = output().getType().cast<RankedTensorType>().getShape();
  with_bias = !bias().getType().isa<mlir::NoneType>();
  relu = do_relu();
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
