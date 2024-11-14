//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/LutFunc.h"

LogicalResult tpu::CumSumOp::init(InferenceParameter &p) { return success(); }

void tpu::CumSumOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::CumSumOp::inference(InferenceParameter &p) {
  auto in_shape = module::getShape(getInput());
  auto dim = getAxis();
  int64_t num_dims = in_shape.size();
  assert(num_dims <= 4);
  assert(dim < in_shape.size());
  int64_t length = in_shape[dim];
  // stride
  int64_t stride = 1;
  for (int64_t i = dim + 1; i < num_dims; i++) {
    stride *= in_shape[i];
  }

  int64_t num_elenments = module::getNumElements(getOutput());
  int64_t cur_index = 0;
  while (cur_index < num_elenments) {
    for (int64_t l = 0; l < length; l++) {
      int64_t start = cur_index + l * stride;
      for (int64_t s = 0; s < stride; s++) {
        if (l == 0) {
          p.outputs[0][start + s] = p.inputs[0][start + s];
        } else {
          p.outputs[0][start + s] =
              p.inputs[0][start + s] + p.outputs[0][start + s - stride];
        }
      }
    }
    cur_index += length * stride;
  }
  return success();
}

bool tpu::CumSumOp::support_multi_core() { return false; }
