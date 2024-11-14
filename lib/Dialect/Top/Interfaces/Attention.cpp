//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Dnnl/Attention.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::AttentionOp::getFLOPs() { return 0; }

LogicalResult top::AttentionOp::init(InferenceParameter &p) {
  auto attention = new Attention();
  auto in_shape = module::getShape(getInput());
  auto key_shape = module::getShape(getKeys());
  auto queries_shape = module::getShape(getQueriesWeight());
  int batch = in_shape[0];
  int M_q = in_shape[1];
  int M_k = key_shape[1];
  int N_q = in_shape[2];
  int N_k = key_shape[2];
  int64_t d = queries_shape[queries_shape.size() - 1] / getHead();
  auto scale = getScale().convertToDouble();

  attention->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.inputs[3],
                   p.inputs[4], p.inputs[5], p.inputs[6], p.inputs[7],
                   p.inputs[8], p.inputs[9], p.inputs[10], p.inputs[11],
                   nullptr, p.outputs[0], nullptr, batch, M_q, M_k, N_q, N_k, d,
                   scale, 0);
  p.handle = (void *)attention;
  return success();
}

void top::AttentionOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto attention = (Attention *)p.handle;
    attention->deinit();
    delete attention;
    p.handle = nullptr;
  }
  return;
}

LogicalResult top::AttentionOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto attention = (Attention *)p.handle;
  attention->run();
  return success();
}

void top::AttentionOp::shape_inference() {}
