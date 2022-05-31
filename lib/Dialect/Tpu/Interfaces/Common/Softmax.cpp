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
#include "sophgo/Support/Helper/Module.h"

using namespace sophgo;
using namespace mlir;

LogicalResult tpu::SoftmaxOp::init(InferenceParameter &p) { return success(); }

void tpu::SoftmaxOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SoftmaxOp::inference(InferenceParameter &p) {
  auto input_shape = Module::getShape(input());
  size_t softmax_cont = input_shape.back();
  size_t batch_cont = Module::getNumElements(input()) / softmax_cont;
  std::vector<float> max_in_each_batch(batch_cont);
  for (size_t i = 0; i < batch_cont; ++i) {
    float max = p.inputs[0][i * softmax_cont];
    for (size_t j = 0; j < softmax_cont; ++j)
      if (p.inputs[0][i * softmax_cont + j] > max)
        max = p.inputs[0][i * softmax_cont + j];
    max_in_each_batch[i] = max;
  }
  std::memcpy(p.outputs[0], p.inputs[0],
              Module::getNumElements(input()) * sizeof(float));
  std::vector<float> exp_sum_in_each_batch(batch_cont, 0);
  for (size_t i = 0; i < batch_cont; ++i) {
    for (size_t j = 0; j < softmax_cont; ++j) {
      auto v = p.outputs[0][i * softmax_cont + j];
      v = std::exp(v - max_in_each_batch[i]);
      p.outputs[0][i * softmax_cont + j] = v;
      exp_sum_in_each_batch[i] += v;
    }
  }
  for (size_t i = 0; i < batch_cont; ++i) {
    for (size_t j = 0; j < softmax_cont; ++j) {
      p.outputs[0][i * softmax_cont + j] /= exp_sum_in_each_batch[i];
    }
  }
  return success();
}
