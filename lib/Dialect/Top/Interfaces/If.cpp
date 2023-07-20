//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::IfOp::getFLOPs() { return 0; }

LogicalResult top::IfOp::init(InferenceParameter &p) { return success(); }
void top::IfOp::deinit(InferenceParameter &p) {}

LogicalResult top::IfOp::inference(InferenceParameter &p) {
  if (p.inputs[0][0] > 0)
    return success(); //then_branch
  else
    return failure(); //else_branch
}

void top::IfOp::shape_inference() {
  auto yield_op = getRegion(0).back().getTerminator();
  std::vector<std::vector<int64_t>> shapes;
  // get shape
  for (auto opd : yield_op->getOperands()) {
    shapes.push_back(module::getShape(opd).vec());
  }
  // check if is vaild
  for (uint32_t i = 1; i < getNumRegions(); i++) {
    yield_op = getRegion(i).back().getTerminator();
    auto nof_inputs = yield_op->getNumOperands();
    assert(nof_inputs == shapes.size() && "Regions have different num of output, fix me.");
    for (uint32_t j = 0; j < nof_inputs; j++) {
      auto _shape = module::getShape(yield_op->getOperand(j)).vec();
      assert((shapes[j] == _shape) && "Regions have different output shape, fix me.");
    }
  }
  // set shape
  for (auto res_shape: llvm::zip(getResults(), shapes)) {
    auto res = std::get<0>(res_shape);
    auto shape = std::get<1>(res_shape);
    module::setShapeOrVerify(res, shape);
  }
  return;
}
