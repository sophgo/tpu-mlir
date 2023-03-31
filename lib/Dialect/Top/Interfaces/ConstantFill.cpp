//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Module.h"
#include "tpu_mlir/Support/MathUtils.h"

int64_t top::ConstantFillOp::getFLOPs() { return 0; }

LogicalResult top::ConstantFillOp::init(InferenceParameter &p) { return success(); }
void top::ConstantFillOp::deinit(InferenceParameter &p) {}

LogicalResult top::ConstantFillOp::inference(InferenceParameter &p) {
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
    module::setShapeOrVerify(getOutput(), shape_);
  } else {
    // do nothing
  }
}
