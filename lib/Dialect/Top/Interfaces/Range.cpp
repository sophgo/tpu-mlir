//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::RangeOp::getFLOPs() { return 0; }

LogicalResult top::RangeOp::init(InferenceParameter &p) { return success(); }
void top::RangeOp::deinit(InferenceParameter &p) {}

LogicalResult top::RangeOp::inference(InferenceParameter &p) {
  float start = module::isNone(getStart()) ? 0 : p.inputs[0][0];
  float delta = module::isNone(getDelta()) ? 1 : p.inputs[2][0];
  auto limit = p.inputs[1][0];
  auto output = p.outputs[0];
  if(limit > start) {
    for (int i = 0, n = start; n < limit; n += delta, ++i)
      output[i] = n;
  } else {
    for (int i = 0, n = start; n > limit; n += delta, ++i)
      output[i] = n;
    }
  return success();
}

// RangeOp is special, will convert to WeightOp
void top::RangeOp::shape_inference() {
  if (!module::isUnranked(getOutput()))
    return;
  int64_t start = 0, delta = 1, limit = 0;
  if (!module::isNone(getStart())) {
    if (auto start_w = dyn_cast<top::WeightOp>(getStart().getDefiningOp())) {
      auto start_v = start_w.read_as_float();
      ASSERT_THIS(start_v->size() == 1);
      start = static_cast<float>(start_v->at(0));
    } else if (module::isShape(getStart())) {
      auto start_v = module::getShapeTensorValue(getStart());
      ASSERT_THIS(start_v.size() == 1);
      start = start_v[0];
    } else {
      llvm_unreachable("start must be a weight or a shape");
    }
  }
  if (!module::isNone(getDelta())) {
    if (auto delta_w = dyn_cast<top::WeightOp>(getDelta().getDefiningOp())) {
      auto delta_v = delta_w.read_as_float();
      ASSERT_THIS(delta_v->size() == 1);
      delta = static_cast<float>(delta_v->at(0));
    } else if (module::isShape(getDelta())) {
      auto delta_v = module::getShapeTensorValue(getDelta());
      ASSERT_THIS(delta_v.size() == 1);
      delta = delta_v[0];
    } else {
      llvm_unreachable("delta must be a weight or a shape");
    }
  }
  if (auto limit_w = dyn_cast<top::WeightOp>(getLimit().getDefiningOp())) {
    auto limit_v = limit_w.read_as_float();
    ASSERT_THIS(limit_v->size() == 1);
    limit = static_cast<float>(limit_v->at(0));
  } else if (module::isShape(getLimit())) {
    auto limit_v = module::getShapeTensorValue(getLimit());
    ASSERT_THIS(limit_v.size() == 1);
    limit = limit_v[0];
  } else {
    llvm_unreachable("limit must be a weight or a shape");
  }
  auto out_size = (limit - start) / delta;
  module::setShapeOrVerify(getOutput(), {out_size});
}
