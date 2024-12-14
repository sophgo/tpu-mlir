//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::FAttentionOp::init(InferenceParameter &p) {
  return success();
}

void tpu::FAttentionOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::FAttentionOp::inference(InferenceParameter &p) {
  auto out_type = module::getStorageType(getOutput());
  bool is_bf16 = out_type.isBF16();
  int batch = getBatch();
  int M_q = getMq();
  int M_k = getMk();
  uint64_t d = getDim();
  uint64_t q_head = getQHead();
  auto kv_head = getKvHead();
  float scale = getScale().convertToDouble();
  int m_size = batch * q_head * M_q * M_k;
  bool has_mask = !module::isNone(getMask());
  auto qk_buffer = new float[m_size];
  auto a16_f = [&](float data) { return is_bf16 ? BF16(data) : F16(data); };
  // Q * K
  dnnl_mm_gqa(p.inputs[0], p.inputs[1], qk_buffer, batch, q_head, kv_head, M_q,
              d, M_k, 0);
  // * scale
  if (!is_bf16) {
    scale = a16_f(scale);
    F16(qk_buffer, qk_buffer, m_size);
  }
#pragma omp parallel for schedule(static, omp_schedule(m_size))
  for (int i = 0; i < m_size; i++) {
    qk_buffer[i] *= scale;
    if (has_mask) {
      int mask_offset = i % (M_q * M_k);
      qk_buffer[i] += p.inputs[3][mask_offset];
    }
  }
  if (!is_bf16) {
    F16(qk_buffer, qk_buffer, m_size);
  }
  // do softmax
  int outer_dim = batch * q_head * M_q;
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; i++) {
    int offset = i * M_k;
    // find max
    float max = is_bf16 ? a16_f(qk_buffer[offset]) : qk_buffer[offset];
    for (int j = 1; j < M_k; j++) {
      float data =
          is_bf16 ? a16_f(qk_buffer[offset + j]) : qk_buffer[offset + j];
      if (max < data) {
        max = data;
      }
    }
    // exp(x- max), sum
    std::vector<float> sub_buffer(M_k, 0.0f);
    float sum = 0;
    for (int j = 0; j < M_k; j++) {
      sub_buffer[j] = a16_f(qk_buffer[offset + j] - max);
      sub_buffer[j] = a16_f(std::exp(sub_buffer[j]));
      sum = sum + sub_buffer[j];
    }
    // divided by sum
    for (int j = 0; j < M_k; j++) {
      qk_buffer[offset + j] = a16_f(sub_buffer[j] * a16_f(1.0f / a16_f(sum)));
    }
  }
  // * V
  float *temp = new float[batch * q_head * M_q * d];
  assert(temp != nullptr);
  dnnl_mm_gqa(qk_buffer, p.inputs[2], temp, batch, q_head, kv_head, M_q, M_k, d,
              1);
  delete[] qk_buffer;
  // * transpose output
  tensor_hc_transpose(p.outputs[0], temp, batch, q_head, M_q, d);
  delete[] temp;

  int out_num = module::getNumElements(getOutput());
  if (is_bf16) {
    BF16(p.outputs[0], p.outputs[0], out_num);
  } else {
    F16(p.outputs[0], p.outputs[0], out_num);
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
