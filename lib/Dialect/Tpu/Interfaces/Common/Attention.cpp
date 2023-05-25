//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Dialect/Tpu/IR/TpuOps.h"
#include "tpu_mlir/Support/Dnnl/Dnnl.h"
#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/Module.h"

#include "tpu_mlir/Backend/BM168x/BM168x.h"
#include "tpu_mlir/Dialect/Tpu/Transforms/Codegen/Dynamic/DynamicLayer.hpp"
#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Dnnl/Attention.h"

const int64_t NPU_NUM = 64;

LogicalResult tpu::AttentionOp::init(InferenceParameter &p) {
  auto attention = new Attention();
  auto in_shape = module::getShape(getInput());
  auto key_shape = module::isNone(getKeys()) ? in_shape : module::getShape(getKeys());
  auto queries_shape = module::getShape(getQueriesWeight());
  auto out_shape = module::getShape(getOutput());
  auto out_type = module::getStorageType(getOutput());
  int batch = in_shape[0];
  int M_q = in_shape[1];
  int M_k = key_shape[1];
  int N_q = in_shape[2];
  int N_k = key_shape[2];
  int64_t d = queries_shape[queries_shape.size() - 1];
  auto scale = getScale().convertToDouble();
  int has_bias = getHasBias();
  auto quant_param = module::getI64Array(getQuantParam());

  float *q_weight = p.inputs[3];
  float *k_weight = q_weight + align_up(in_shape[2], NPU_NUM) * d;
  float *v_weight = k_weight + align_up(key_shape[2], NPU_NUM) * d;
  float *bias_offset = p.inputs[4];
  float *q_bias = has_bias&0x01 ? bias_offset : nullptr;
  int len = has_bias&0x01 ? d : 0;
  float *k_bias = has_bias&0x02 ? bias_offset + len : nullptr;
  len += has_bias&0x02 ? d : 0;
  float *v_bias = has_bias&0x04 ? bias_offset + len : nullptr;
  len += has_bias&0x04 ? d : 0;
  float *o_bias = has_bias&0x08 ? bias_offset + len : nullptr;
  len += has_bias&0x08 ? N_q : 0;
  // float *table = has_bias&0x10 ? bias_offset + len : nullptr;
  int type = out_type.isF16() ? 1 : 0;
  type = out_type.isBF16() ? 2 : type;
  type = out_type.isInteger(32) ? 3 : type;

  attention->setup(p.inputs[0], p.inputs[1], p.inputs[2], q_weight, q_bias,
                   k_weight, k_bias, v_weight, v_bias, p.inputs[9],
                   o_bias, p.inputs[11], p.inputs[12], p.outputs[0], quant_param->data(),
                   batch, M_q, M_k, N_q, N_k,
                   d, scale, 0, type);
  p.handle = (void *)attention;
  return success();
}

void tpu::AttentionOp::deinit(InferenceParameter &p) {
  if (p.handle != nullptr) {
    auto attention = (Attention *)p.handle;
    attention->deinit();
    delete attention;
    p.handle = nullptr;
  }
  return;
}

LogicalResult tpu::AttentionOp::inference(InferenceParameter &p) {
  if (p.handle == nullptr) {
    return failure();
  }
  auto attention = (Attention *)p.handle;
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

LogicalResult tpu::AttentionOp::LocalGenSupport() {
  if (module::isCV18xx()) {
    return failure();
  }
  return success();
}

LogicalResult tpu::AttentionOp::AllowDataSplit(int64_t axis,
                                            group_type_t group_type) {
  if (axis == 0) {
    return success();
  } else if (axis == 1) {
    if (!module::isNone(getKeys())) {
      return success();
    }
  }
  return failure();
}

mlir::Type tpu::AttentionOp::type_verify(uint64_t opd_idx, TypeCastMode &mode) {
  if (opd_idx == 0 || opd_idx == 1) {
    return type_verify_case_i32(getOperation(), opd_idx, mode);
  }
  return type_verify_case_same(getOperation(), opd_idx, mode);
}

// void tpu::AttentionOp::assign_fw_param(void *param) {

// }
