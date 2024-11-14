//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"

int64_t top::DepackRawOp::getFLOPs() {
  return module::getNumElements(getOutput());
}

LogicalResult top::DepackRawOp::init(InferenceParameter &p) {
  return success();
}
void top::DepackRawOp::deinit(InferenceParameter &p) {}

LogicalResult top::DepackRawOp::inference(InferenceParameter &p) {
  return success();
}

void top::DepackRawOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  int block_size[2] = {2, 2};
  int bh = block_size[0];
  int bw = block_size[1];

  int ph = getPaddingH();
  int pw = getPaddingW();

  int ic = in_shape[1];
  int ih = in_shape[2] - ph;
  int iw = in_shape[3] - pw;
  assert(ic == bh * bw);
  int oh = ih * bh;
  int ow = iw * bw;

  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(in_shape[0]);
  out_shape.push_back(1);
  out_shape.push_back(oh);
  out_shape.push_back(ow);
  module::setShapeOrVerify(getOutput(), out_shape);
}
