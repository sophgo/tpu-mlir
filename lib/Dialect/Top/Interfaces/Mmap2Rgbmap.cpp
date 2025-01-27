//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::Mmap2RgbmapOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::Mmap2RgbmapOp::init(InferenceParameter &p) {
  return success();
}
void top::Mmap2RgbmapOp::deinit(InferenceParameter &p) {}

LogicalResult top::Mmap2RgbmapOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::Mmap2RgbmapOp::shape_inference() {
  auto in_shape = module::getShape(getInput());

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(in_shape[0]);
  out_shape.push_back(in_shape[1]);
  out_shape.push_back(in_shape[2]);
  out_shape.push_back(in_shape[3] * 6);
  module::setShapeOrVerify(getOutput(), out_shape);
}
