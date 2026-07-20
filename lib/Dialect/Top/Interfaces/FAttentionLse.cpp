//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "tpu_mlir/Support/MathUtils.h"
#include "tpu_mlir/Support/Module.h"

int64_t top::FAttentionLseOp::getFLOPs() {
  int batch = getBatch();
  int M_q = getMq();
  int M_k = getMk();
  uint64_t d = getDim();
  uint64_t q_head = getQHead();
  return batch * M_q * q_head * d * M_k * 4;
}

LogicalResult top::FAttentionLseOp::init(InferenceParameter &p) {
  return success();
}
void top::FAttentionLseOp::deinit(InferenceParameter &p) {}

// Identical to FAttentionOp::inference but also captures the per-row max/sum
// and writes lse = max + log(sum) to outputs[1]. The kernel stores lse in
// [b, mq, q_head] order (mq before q_head, matching the attention output);
// the softmax loop iterates i over [b, q_head, mq], so we transpose when
// writing lse.
LogicalResult top::FAttentionLseOp::inference(InferenceParameter &p) {
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
  dnnl_mm_gqa(p.inputs[0], p.inputs[1], qk_buffer, batch, q_head, kv_head, M_q,
              d, M_k, 0);
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
  int outer_dim = batch * q_head * M_q;
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int i = 0; i < outer_dim; i++) {
    int offset = i * M_k;
    float max = qk_buffer[offset];
    for (int j = 1; j < M_k; j++) {
      float data = qk_buffer[offset + j];
      if (max < data) {
        max = data;
      }
    }
    std::vector<float> sub_buffer(M_k, 0.0f);
    float sum = 0;
    for (int j = 0; j < M_k; j++) {
      sub_buffer[j] = qk_buffer[offset + j] - max;
      sub_buffer[j] = std::exp(sub_buffer[j]);
      sum = sum + sub_buffer[j];
    }
    for (int j = 0; j < M_k; j++) {
      qk_buffer[offset + j] = sub_buffer[j] * (1.0f / sum);
    }
    // lse = max + log(sum). Loop i is over [b, q_head, M_q]; write to lse
    // output laid out as [b, M_q, q_head] (transpose the q_head/M_q axes).
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

  return success();
}

// output: same as FAttention. lse: [b, M_q, q_head] (keep_dims=false) or
// [b, M_q, q_head, 1] (keep_dims=true). NOTE: cannot use common_shape_inference
// (it asserts a single output); set both result shapes explicitly.
void top::FAttentionLseOp::shape_inference() {
  auto out = getOutput();
  bool keep_dims = getKeepDims();
  auto in_shape = module::getShape(getQueries());
  assert(in_shape.size() == 4);
  if (keep_dims) {
    module::setShapeOrVerify(out, in_shape);
    module::setShapeOrVerify(getLse(),
                             {in_shape[0], in_shape[1], in_shape[2], 1});
    return;
  }
  std::vector<int64_t> out_shape;
  out_shape.push_back(in_shape[0]);
  out_shape.push_back(in_shape[1]);
  out_shape.push_back(in_shape[2] * in_shape[3]);
  module::setShapeOrVerify(out, out_shape);
  // lse: flatten q_head * 1 (last two dims) -> [b, M_q, q_head]
  std::vector<int64_t> lse_shape = {in_shape[0], in_shape[1], in_shape[2]};
  module::setShapeOrVerify(getLse(), lse_shape);
}
