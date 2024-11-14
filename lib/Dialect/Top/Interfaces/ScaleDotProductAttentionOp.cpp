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

int64_t top::ScaleDotProductAttentionOp::getFLOPs() {
  auto query_op = getQuery();
  auto key_op = getKey();
  auto value_op = getValue();
  auto query_sizes = module::getNumElements(query_op);
  auto keys = module::getNumElements(key_op);
  auto values = module::getNumElements(value_op);
  auto softmax_flops = values * 5;
  auto div_op_flops = values;
  return query_sizes * keys * values + softmax_flops + div_op_flops;
}

LogicalResult top::ScaleDotProductAttentionOp::init(InferenceParameter &p) {
  return success();
}

void top::ScaleDotProductAttentionOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto attention = (ScaledDotProductAttention *)p.handle;
    attention->deinit();
    delete attention;
    p.handle = nullptr;
  }
  return;
}

LogicalResult
top::ScaleDotProductAttentionOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto attention = (ScaledDotProductAttention *)p.handle;
  attention->run();
  return success();
}

void top::ScaleDotProductAttentionOp::shape_inference() {
  std::vector<int64_t> out_shape;
  auto query_shape = module::getShape(getQuery());
  auto value_shape = module::getShape(getValue());
  auto shape_len = query_shape.size();
  int64_t batch = query_shape[0];
  out_shape.push_back(batch);
  if (shape_len == 4) {
    out_shape.push_back(query_shape[1]);
  }
  int64_t query_len = query_shape[shape_len - 2];
  out_shape.push_back(query_len);
  int64_t value_dim = value_shape[shape_len - 1];
  out_shape.push_back(value_dim);
  auto out = getOutput();
  module::setShapeOrVerify(out, out_shape);
}
