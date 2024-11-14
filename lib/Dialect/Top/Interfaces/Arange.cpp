//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ArangeOp::getFLOPs() { return 0; }

LogicalResult top::ArangeOp::init(InferenceParameter &p) { return success(); }
void top::ArangeOp::deinit(InferenceParameter &p) {}

LogicalResult top::ArangeOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

// ArangeOp is special, will convert to WeightOp
void top::ArangeOp::shape_inference() {
  float start = 0;
  float step = 1;
  float end = 0;
  if (auto end_w = dyn_cast<top::WeightOp>(getEnd().getDefiningOp())) {
    auto end_v = end_w.read_as_float();
    ASSERT_THIS(end_v->size() == 1);
    end = static_cast<float>(end_v->at(0));
  } else if (module::isShape(getEnd())) {
    auto end_v = module::getShapeTensorValue(getEnd());
    ASSERT_THIS(end_v.size() == 1);
    end = end_v[0];
  } else if (auto end_min_op = dyn_cast<top::MinOp>(getEnd().getDefiningOp())) {
    if (auto end_min_w = dyn_cast<top::WeightOp>(
            end_min_op.getInputs()[1].getDefiningOp())) {
      auto end_v = end_min_w.read_as_float();
      end = static_cast<float>(end_v->at(0));
    } else {
      llvm_unreachable("End is tensor");
    }
  } else {
    llvm_unreachable("End must be a weight or a shape");
  }

  if (module::isNone(getStart()) == false) {
    if (module::isWeight(getStart())) {
      auto start_op = getStart().getDefiningOp<top::WeightOp>();
      auto start_data = start_op.read<float>();
      start = start_data->at(0);
    } else if (auto start_max_op =
                   dyn_cast<top::MaxOp>(getStart().getDefiningOp())) {
      if (auto start_max_w = dyn_cast<top::WeightOp>(
              start_max_op.getInputs()[1].getDefiningOp())) {
        auto start_v = start_max_w.read_as_float();
        start = static_cast<float>(start_v->at(0));
      } else {
        llvm_unreachable("start is tensor");
      }
    }
  }
  if (module::isNone(getStep()) == false) {
    ASSERT_THIS(module::isWeight(getStep()));
    auto step_op = getStep().getDefiningOp<top::WeightOp>();
    auto step_data = step_op.read<float>();
    step = step_data->at(0);
    ASSERT_THIS(step != 0);
  }
  std::vector<float> data;
  for (int i = start; i < end; i += step) {
    data.push_back(i);
  }
  auto op = getOperation();
  OpBuilder builder(module::getCtx());
  builder.setInsertionPointAfter(op);
  auto weight_type =
      RankedTensorType::get({(int64_t)data.size()}, builder.getF32Type());
  auto new_op = top::WeightOp::create(op, "arange", data, weight_type);
  getOutput().replaceAllUsesWith(new_op);
}
