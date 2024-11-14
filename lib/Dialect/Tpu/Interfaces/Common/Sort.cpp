//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::SortOp::init(InferenceParameter &p) { return success(); }
void tpu::SortOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::SortOp::inference(InferenceParameter &p) {
  sort_param_t param = {.axis = (int)getAxis(), .descending = getDescending()};
  auto shape = module::getShape(getInput());
  int dims = shape.size();
  std::vector<int> shape_v(dims);
  for (int i = 0; i < dims; ++i) {
    shape_v[i] = shape[i];
  }
  sort_per_dim(param, shape_v.data(), dims, p.inputs[0], p.outputs[0],
               p.outputs[1]);
  return success();
}

mlir::Type tpu::SortOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  (void)(opd_idx);
  if (module::isNone(getValues()))
    return do_nothing(mode);
  else {
    auto op = getOperation();
    return type_verify_case_same(op, 0, mode);
  }
}

bool tpu::SortOp::support_multi_core() { return false; }
