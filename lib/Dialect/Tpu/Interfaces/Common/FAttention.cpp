//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"

#include "tpu_mlir/Support/Dnnl/FAttention.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::FAttentionOp::init(InferenceParameter &p) {
  auto attention = new FAttention();
  auto out_type = module::getStorageType(getOutput());
  int batch = getBatch();
  int M_q = getMq();
  int M_k = getMk();
  uint64_t d = getDim();
  uint64_t head = getQHead();
  auto scale = getScale().convertToDouble();

  int type = out_type.isF16() ? 1 : 0;
  type = out_type.isBF16() ? 2 : type;
  type = out_type.isInteger(32) ? 3 : type;

  attention->setup(p.inputs[0], p.inputs[1], p.inputs[2], p.inputs[3],
                   p.outputs[0], batch, M_q, M_k, head, d, scale, type);
  p.handle = (void *)attention;
  return success();
}

void tpu::FAttentionOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto attention = (FAttention *)p.handle;
    attention->deinit();
    delete attention;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::FAttentionOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto attention = (FAttention *)p.handle;
  auto out_type = module::getStorageType(getOutput());
  auto num_elem = module::getNumElements(getOutput());
  attention->run();
  if (out_type.isBF16()) {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  } else if (out_type.isF16()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  }
  return success();
}

mlir::Type tpu::FAttentionOp::type_verify(uint64_t opd_idx,
                                          TypeCastMode &mode) {
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

// void tpu::FAttentionOp::assign_fw_param(void *param) {

// }

bool tpu::FAttentionOp::support_multi_core() { return true; }
