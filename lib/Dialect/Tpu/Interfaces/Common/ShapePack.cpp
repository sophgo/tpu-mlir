//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Backend/CV18xx/CV18xx.h"
#include "tpu_mlir/Support/Dnnl/Concat.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"

using namespace tpu_mlir::backend;

LogicalResult tpu::ShapePackOp::init(InferenceParameter &p) {
  auto concat = new Concat();
  auto axis_ = getAxis();
  concat_attr_t attr;
  attr.num_src = getInputs().size();

  for (int i = 0; i < attr.num_src; i++) {
    auto input_shape = module::getShape(getInputs()[i]);
    int channel = input_shape[axis_];

    int outer_dim = 1;
    for (int i = 0; i < axis_; i++) {
      outer_dim *= input_shape[i];
    }
    int inner_dim = 1;
    for (int i = axis_ + 1; i < input_shape.size(); i++) {
      inner_dim *= input_shape[i];
    }

    attr.src_shapes.push_back({outer_dim, channel, inner_dim});
  }

  attr.dst_shape = module::getShape(getOutput());
  attr.axis = 1;
  concat->setup(p.inputs, p.outputs[0], attr);
  p.handle = (void *)concat;
  return success();
}
void tpu::ShapePackOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto concat = (Concat *)(p.handle);
    delete concat;
    p.handle = nullptr;
  }
}

LogicalResult tpu::ShapePackOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto concat = (Concat *)p.handle;
  concat->run();
  return success();
}

bool tpu::ShapePackOp::support_multi_core() { return false; }
