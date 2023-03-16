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

int64_t top::ZerosOp::getFLOPs() { return 0; }

LogicalResult top::ZerosOp::init(InferenceParameter &p) { return success(); }
void top::ZerosOp::deinit(InferenceParameter &p) {}

LogicalResult top::ZerosOp::inference(InferenceParameter &p) {
  auto num_elem = module::getNumElements(getOutput());
  memset(p.outputs[0], 0, sizeof(float) * num_elem);
  return success();
}

void top::ZerosOp::shape_inference() {
  assert(module::isWeight(getInput()));
  auto weight = cast<top::WeightOp>(getInput().getDefiningOp());
  auto shape = weight.read<float>();
  std::vector<int64_t> shape_(shape->begin(), shape->end());
  module::setShapeOrVerify(getOutput(), shape_);
}
