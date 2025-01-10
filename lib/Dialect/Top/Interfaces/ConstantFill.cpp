//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ConstantFillOp::getFLOPs() { return 0; }

LogicalResult top::ConstantFillOp::init(InferenceParameter &p) {
  return success();
}
void top::ConstantFillOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConstantFillOp::inference(InferenceParameter &p) {
  if (module::isWeight(getInput())) {
    auto weight = cast<top::WeightOp>(getInput().getDefiningOp());
    const auto shape = weight.read<float>();
    std::vector<int64_t> shape_(shape->begin(), shape->end());
    int idx = 0;
    for (auto a : shape_) {
      if (a == -1)
        shape_[idx] = (int64_t)1;
      idx += 1;
    }
    module::setShape(getOutput(), shape_);
  } else {
    auto num_elem = module::getNumElements(getInput());
    std::vector<int64_t> out_shape;
    for (int i = 0; i < num_elem; i++) {
      out_shape.push_back(p.inputs[0][i]);
    }
    module::setShape(getOutput(), out_shape);
  }
  auto num_elem = module::getNumElements(getOutput());
  float const_val = getValue().convertToDouble();
#pragma omp parallel for schedule(static, omp_schedule(num_elem))
  for (int i = 0; i < num_elem; ++i) {
    p.outputs[0][i] = const_val;
  }
  return success();
}

void top::ConstantFillOp::shape_inference() {
  if (module::isWeight(getInput())) {
    auto weight = cast<top::WeightOp>(getInput().getDefiningOp());
    const auto shape = weight.read<float>();
    std::vector<int64_t> shape_(shape->begin(), shape->end());
    int idx = 0;
    for (auto a : shape_) {
      if (a == -1)
        shape_[idx] = (int64_t)1;
      idx += 1;
    }
    module::setShapeOrVerify(getOutput(), shape_);
  } else if (module::isShape(getInput())) {
    auto out_shape = module::getShapeTensorValue(getInput());
    module::setShapeOrVerify(getOutput(), out_shape);
    if (out_shape.size() == 1 || out_shape.size() == 0) {
      std::vector<std::vector<int64_t>> input_shapes_v;
      input_shapes_v.push_back(out_shape);
      auto output_shape_v = module::commonShapeValInfer(
          getOperation(), input_shapes_v, out_shape);
      module::bindShapeTensorValue(getOutput(), output_shape_v);
    }
  } else {
    auto output_shape = module::getShape(getInput());
    module::setShapeOrVerify(getOutput(), output_shape);
  }
}
