//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Concat.h"
#include "tpu_mlir/Dialect/Top/IR/TopOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Helper/Module.h"

using namespace tpu_mlir;
using namespace tpu_mlir::helper;
using namespace mlir;

int64_t top::ConcatOp::getFLOPs() { return 0; }

LogicalResult top::ConcatOp::init(InferenceParameter &p) {
  auto concat = new Concat();
  p.handle = (void *)concat;
  return success();
}
void top::ConcatOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto concat = (Concat *)(p.handle);
    delete concat;
    p.handle = nullptr;
  }
}

LogicalResult top::ConcatOp::inference(InferenceParameter &p) {
  auto axis_ = axis();
  auto input_shape = inputs()[0].getType().cast<RankedTensorType>().getShape();
  concat_attr_t attr;
  int channel = input_shape[axis_];

  int outer_dim = 1;
  for (int i = 0; i < axis_; i++) {
    outer_dim *= input_shape[i];
  }
  int inner_dim = 1;
  for (int i = axis_ + 1; i < input_shape.size(); i++) {
    inner_dim *= input_shape[i];
  }

  attr.src_shape = {outer_dim, channel, inner_dim};
  attr.dst_shape = attr.src_shape;
  attr.axis = axis_;

  auto concat = (Concat *)p.handle;
  concat->setup(p.inputs, p.outputs[0], attr);
  concat->run();
  return success();
}
