//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ReshapeOp::getFLOPs() { return 0; }

LogicalResult top::ReshapeOp::init(InferenceParameter &p) { return success(); }
void top::ReshapeOp::deinit(InferenceParameter &p) {}

LogicalResult top::ReshapeOp::inference(InferenceParameter &p) {
  if (p.inputs[0] != p.outputs[0]) {
    auto num_elem = module::getNumElements(getOutput());
    memcpy(p.outputs[0], p.inputs[0], num_elem * sizeof(float));
  }
  return success();
}

void top::ReshapeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num = module::getNumElements(getInput());
  std::vector<int64_t> out_shape;
  if (getShape().has_value()) {
    auto shape = module::getI64Array(getShape().value());
    int x = -1;
    for (int i = 0; i < shape->size(); i++) {
      auto s = shape->at(i);
      if (s > 0) {
        out_shape.push_back(s);
        num /= s;
      } else if (s == 0) {
        out_shape.push_back(in_shape[i]);
        num /= in_shape[i];
      } else if (s == -1) {
        out_shape.push_back(-1);
        x = i;
      } else {
        dump();
        llvm_unreachable("shape is illegal");
      }
    }
    if (x >= 0) {
      out_shape[x] = num;
    }
    module::setShapeOrVerify(getOutput(), out_shape);
  } else if (getShapeT()) {
    if (auto shape_w = dyn_cast<top::WeightOp>(getShapeT().getDefiningOp())) {
      auto shape_v = shape_w.read_as_float();
      std::transform(shape_v->begin(), shape_v->end(),
                     std::back_inserter(out_shape),
                     [](auto &v) { return static_cast<int64_t>(v); });
    } else if (module::isShape(getShapeT())) {
      out_shape = module::getShapeTensorValue(getShapeT());
    } else {
      llvm_unreachable("shape is illegal");
    }
    assert(std::count(out_shape.begin(), out_shape.end(), -1) <= 1);
    auto last_0_iter = std::find(out_shape.rbegin(), out_shape.rend(), 0);
    auto last_0_bias = std::distance(last_0_iter, out_shape.rend());
    assert(last_0_bias <= in_shape.size());
    std::transform(out_shape.begin(), out_shape.begin() + last_0_bias,
                   in_shape.begin(), out_shape.begin(),
                   [](auto out, auto in) { return out == 0 ? in : out; });
    auto unrank_shape_iter = std::find(out_shape.begin(), out_shape.end(), -1);
    if (unrank_shape_iter != out_shape.end()) {
      auto fixed_shape_elem = std::accumulate(
          out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());
      *unrank_shape_iter = -num / fixed_shape_elem;
    }
    module::setShapeOrVerify(getOutput(), out_shape);
  } else {
    // for tflite, no shape input or attribute
    auto out_shape = module::getShape(getOutput());
    module::setShapeOrVerify(getOutput(), out_shape);
  }

  if (!module::isUnranked(getOutput())) {
    auto num_input = module::getNumElements(getInput());
    auto num_output = module::getNumElements(getOutput());
    assert(num_input == num_output);
  }

  if (module::isShape(getInput())) {
    auto out_shape = module::getShapeTensorValue(getInput());
    module::bindShapeTensorValue(getOutput(), out_shape);
  }

  removeShapeAttr();
}
