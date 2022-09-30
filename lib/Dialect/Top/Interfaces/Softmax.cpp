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
#include "tpu_mlir/Support/Helper/Module.h"
#include "tpu_mlir/Support/Dnnl/Softmax.h"

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
  p.handle = (void*)softmax;
  return success();
}
void top::SoftmaxOp::deinit(InferenceParameter &p) {
  if(p.handle != nullptr) {
    auto softmax = (Softmax*)(p.handle);
    delete softmax;
    p.handle = nullptr;
  }
}

LogicalResult top::SoftmaxOp::inference(InferenceParameter &p) {
  if(p.handle == nullptr) {
    return failure();
  }
  auto axis_ = axis();
  auto input_shape = Module::getShape(input());
  int channel = input_shape[axis_];

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }
  int inner_dim = 1;
  for (int i = axis_ + 1; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  softmax_attr_t attr;
  attr.axis = axis_;
  attr.src_shape = {inner_dim, channel};
  attr.dst_shape = {outer_dim, channel};

  auto softmax = (Softmax*)p.handle;
  softmax->setup(p.inputs[0], p.outputs[0], attr);
  softmax->run();
  return success();
}