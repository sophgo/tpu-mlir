//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Dnnl.h"

int64_t top::LRNOp::getFLOPs() {
  int64_t n, c, h, w;
  module::getNCHW(getInput(), n, c, h, w);
  return module::getNumElements(getInput()) *
         (5 /*eltwise gops*/ +
          (c - getSize()) /*fully reduce sum*/ * (getSize() - 1) /*sum gops*/ -
          getSize() /*fix edge split*/);
}

LogicalResult top::LRNOp::init(InferenceParameter &p) {
  auto lrn = new LRN();
  (*lrn)
      .src(p.inputs[0], module::getShape(getInput()))
      .dst(p.outputs[0], module::getShape(getOutput()))
      .size(getSize())
      .param(getAlpha().convertToDouble(), getBeta().convertToDouble(),
             getBias().convertToDouble())
      .algorithem(algorithm::lrn_across_channels)
      .setup();

  p.handle = (void *)lrn;
  return success();
}
void top::LRNOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto lrn = (LRN *)p.handle;
    delete lrn;
    p.handle = nullptr;
  }
}

LogicalResult top::LRNOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr)
    return failure();
  auto lrn = (LRN *)p.handle;
  lrn->run();
  return success();
}

void top::LRNOp::shape_inference() { common_shape_inference(getOperation()); }
