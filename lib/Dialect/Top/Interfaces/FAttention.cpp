//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
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
  return success();
}
void top::FAttentionOp::deinit(InferenceParameter &p) {}

LogicalResult top::FAttentionOp::inference(InferenceParameter &p) {
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
  // Q * K
  dnnl_mm_gqa(p.inputs[0], p.inputs[1], qk_buffer, batch, q_head, kv_head, M_q,
              d, M_k, 0);
  // * scale
#pragma omp parallel for schedule(static, omp_schedule(m_size))
  for (int i = 0; i < m_size; i++) {
    qk_buffer[i] *= scale;
    if (has_mask) {
      int mask_offset = i % (M_q * M_k);
      qk_buffer[i] += p.inputs[3][mask_offset];
    }
  }
  // do softmax
  int outer_dim = batch * q_head * M_q;
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; i++) {
    int offset = i * M_k;
    // find max
    float max = qk_buffer[offset];
    for (int j = 1; j < M_k; j++) {
      float data = qk_buffer[offset + j];
      if (max < data) {
        max = data;
      }
    }
    // exp(x- max), sum
    std::vector<float> sub_buffer(M_k, 0.0f);
    float sum = 0;
    for (int j = 0; j < M_k; j++) {
      sub_buffer[j] = qk_buffer[offset + j] - max;
      sub_buffer[j] = std::exp(sub_buffer[j]);
      sum = sum + sub_buffer[j];
    }
    // divided by sum
    for (int j = 0; j < M_k; j++) {
      qk_buffer[offset + j] = sub_buffer[j] * (1.0f / sum);
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

  return success();
}

void top::FAttentionOp::shape_inference() {
  UNREACHABLE_THIS("Not Implemented");
}
