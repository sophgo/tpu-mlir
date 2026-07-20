//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/Float16.h"
#include "tpu_mlir/Support/MathUtils.h"

LogicalResult tpu::FAttentionLseOp::init(InferenceParameter &p) {
  return success();
}

void tpu::FAttentionLseOp::deinit(InferenceParameter &p) {}

// Mirrors tpu::FAttentionOp::inference plus a fp32 lse output (outputs[1]).
LogicalResult tpu::FAttentionLseOp::inference(InferenceParameter &p) {
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
  int mask_size = getMaskSize();
  bool full_mask = (mask_size == 0);
  int q_pos_offset = M_k - M_q;
  auto qk_buffer = new float[m_size];
  auto a16_f = [&](float data) { return is_bf16 ? BF16(data) : F16(data); };
  dnnl_mm_gqa(p.inputs[0], p.inputs[1], qk_buffer, batch, q_head, kv_head, M_q,
              d, M_k, 0);
  if (!is_bf16) {
    scale = a16_f(scale);
    F16(qk_buffer, qk_buffer, m_size);
  }
#pragma omp parallel for schedule(static, omp_schedule(m_size))
  for (int i = 0; i < m_size; i++) {
    qk_buffer[i] *= scale;
    if (has_mask) {
      if (full_mask) {
        int mask_offset = i % (M_q * M_k);
        qk_buffer[i] += p.inputs[3][mask_offset];
      } else {
        int mk = i % M_k;
        int mq = (i / M_k) % M_q;
        if (mk > mq + q_pos_offset) {
          qk_buffer[i] = -std::numeric_limits<float>::infinity();
        }
      }
    }
  }
  if (!is_bf16) {
    F16(qk_buffer, qk_buffer, m_size);
  }
  int outer_dim = batch * q_head * M_q;
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; i++) {
    int offset = i * M_k;
    float max = is_bf16 ? a16_f(qk_buffer[offset]) : qk_buffer[offset];
    for (int j = 1; j < M_k; j++) {
      float data =
          is_bf16 ? a16_f(qk_buffer[offset + j]) : qk_buffer[offset + j];
      if (max < data) {
        max = data;
      }
    }
    std::vector<float> sub_buffer(M_k, 0.0f);
    float sum = 0;
    for (int j = 0; j < M_k; j++) {
      sub_buffer[j] = a16_f(qk_buffer[offset + j] - max);
      sub_buffer[j] = a16_f(std::exp(sub_buffer[j]));
      sum = sum + sub_buffer[j];
    }
    for (int j = 0; j < M_k; j++) {
      qk_buffer[offset + j] = a16_f(sub_buffer[j] * a16_f(1.0f / a16_f(sum)));
    }
    // lse = max + log(sum); loop i is [b, q_head, M_q], output lse is
    // [b, M_q, q_head] (transpose q_head/M_q).
    int b_idx = i / (q_head * M_q);
    int rem = i % (q_head * M_q);
    int qh_idx = rem / M_q;
    int mq_idx = rem % M_q;
    int lse_offset = b_idx * (M_q * q_head) + mq_idx * q_head + qh_idx;
    p.outputs[1][lse_offset] = max + std::log(sum);
  }
  float *temp = new float[batch * q_head * M_q * d];
  assert(temp != nullptr);
  dnnl_mm_gqa(qk_buffer, p.inputs[2], temp, batch, q_head, kv_head, M_q, M_k, d,
              1);
  delete[] qk_buffer;
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

mlir::Type tpu::FAttentionLseOp::type_verify(uint64_t opd_idx,
                                             TypeCastMode &mode) {
  auto mask_size = getMaskSize();
  if (opd_idx == 3 && mask_size > 0) {
    auto dtype = module::getStorageType(getMask());
    if (!dtype.isF32()) {
      mode = TypeCastMode::DO_CAST;
      return mlir::Float32Type::get(getContext());
    } else {
      return do_nothing(mode);
    }
  } else {
    return type_verify_case_same(getOperation(), opd_idx, mode);
  }
}

bool tpu::FAttentionLseOp::support_multi_core() { return true; }
