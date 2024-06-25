//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::Yuv2rgbFormulaOp::getFLOPs() { return 0; }

LogicalResult top::Yuv2rgbFormulaOp::init(InferenceParameter &p) {
  return success();
}
void top::Yuv2rgbFormulaOp::deinit(InferenceParameter &p) {}

LogicalResult top::Yuv2rgbFormulaOp::inference(InferenceParameter &p) {
  llvm_unreachable("Not Implemented");
}

void top::Yuv2rgbFormulaOp::shape_inference() {
  auto YUV_shape = module::getShape(getYUV());
  auto out_shape = llvm::SmallVector<int64_t>(YUV_shape);
  assert(out_shape.size() >= 2);
  out_shape.insert(out_shape.end() - 2, 3);
  out_shape[out_shape.size() - 2] = out_shape[out_shape.size() - 2] / 3 * 2;
  auto out = getOutput();
  module::setShapeOrVerify(out, out_shape);
}