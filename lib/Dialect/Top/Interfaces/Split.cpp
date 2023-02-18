//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"


bool top::SplitOp::isEltwise() {
  return false;
}

int64_t top::SplitOp::getFLOPs() {
  return 0;
}

LogicalResult top::SplitOp::init(InferenceParameter &p) { return success(); }
void top::SplitOp::deinit(InferenceParameter &p) {}

LogicalResult top::SplitOp::inference(InferenceParameter &p) {
  // auto out_num_elem = module::getNumElements(getOutput());
  int out_num = getNum();
  int split_axis = getAxis();
  auto in_shape = module::getShape(getInput());
  int64_t out_max_size = (in_shape[split_axis] + out_num - 1) / out_num;
  std::vector<int64_t>out_size(out_num);
  for (int i = 0; i < out_num - 1; ++i) {
    out_size[i] = out_max_size;
  }
  out_size[out_num - 1] = in_shape[split_axis] - (out_num - 1) * out_max_size;
  int64_t outer_num_elem = 1, inner_num_elem = 1;
  for (int i = 0; i < split_axis; ++i) {
    outer_num_elem *= in_shape[i];
  }
  for (int i = split_axis + 1; i < in_shape.size(); ++i) {
    inner_num_elem *= in_shape[i];
  }

#pragma omp parallel for schedule(static, omp_schedule(outer_num_elem))
  for (int i = 0; i < outer_num_elem; ++i) {
    int64_t index = i * in_shape[split_axis] * inner_num_elem;
    for (int j = 0; j < out_num; ++j) {
      memcpy(p.outputs[j] + i * out_size[j] * inner_num_elem,
             p.inputs[0] + index + j * out_max_size * inner_num_elem,
             out_size[j] * inner_num_elem * sizeof(float));
    }
  }

  return success();
}
