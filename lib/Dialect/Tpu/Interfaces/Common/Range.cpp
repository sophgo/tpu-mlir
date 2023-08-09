//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::RangeOp::init(InferenceParameter &p) { return success(); }
void tpu::RangeOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RangeOp::inference(InferenceParameter &p) {
  auto start = p.inputs[0][0];
  auto limit = p.inputs[1][0];
  auto delta = p.inputs[2][0];
  auto output = p.outputs[0];
  for (int i = 0, n = start; n < limit; n += delta, ++i)
    output[i] = n;

  return success();
}
