//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Softmax.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::SoftmaxOp::getFLOPs() {
  //   2*n          -- compute shifted logits
  //   n            -- exp of shifted logits
  //   2*n          -- compute softmax from exp of shifted logits
  return Module::getNumElements(input()) * 5;
}

LogicalResult top::SoftmaxOp::init(InferenceParameter &p) {
  auto softmax = new Softmax();
  p.handle = (void *)softmax;
  return success();
}
void top::SoftmaxOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto softmax = (Softmax *)(p.handle);
    delete softmax;
    p.handle = nullptr;
  }
}

LogicalResult top::SoftmaxOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto axis_ = axis();
  auto input_shape = Module::getShape(input());
  softmax_attr_t attr;
  attr.axis = axis_;
  attr.src_shape = input_shape;
  attr.dst_shape = input_shape;

  auto softmax = (Softmax *)p.handle;
  softmax->setup(p.inputs[0], p.outputs[0], attr);
  softmax->run();
  return success();
}
