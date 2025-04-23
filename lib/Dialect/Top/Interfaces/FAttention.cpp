//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Module.h"

int64_t top::FAttentionOp::getFLOPs() {
  int batch = getBatch();
  int M_q = getMq();
  int M_k = getMk();
  uint64_t d = getDim();
  uint64_t q_head = getQHead();
  // [batch, M_q, q_head, d] * [batch, M_k, kv_head, d] => [batch, M_q, q_head,
  // M_k]
  // [batch, M_q, q_head, M_k] * [batch, M_k, kv_head, d] => [batch, M_q,
  // q_head, d]
  return batch * M_q * q_head * d * M_k * 4;
}

LogicalResult top::FAttentionOp::init(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}
void top::FAttentionOp::deinit(InferenceParameter &p) {}

LogicalResult top::FAttentionOp::inference(InferenceParameter &p) {
  UNREACHABLE_THIS("Not Implemented");
  return success();
}

void top::FAttentionOp::shape_inference() {
  UNREACHABLE_THIS("Not Implemented");
}
