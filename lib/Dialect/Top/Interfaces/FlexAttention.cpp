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

// CPU reference for FlexAttention. Mirrors FAttentionLseOp::inference with the
// qk_d/v_d split: QK contracts over qk_d, AV produces v_d. The block_bitmap is
// a kernel compute optimisation only -- a skipped (fully-masked) block
// contributes exp(-inf)=0 and does not change the row max, which is exactly
// what the additive mask already encodes, so the CPU ref ignores it. (Caveat:
// a query row with NO unmasked key diverges -- kernel leaves mi=-inf/li=0 ->
// NaN output; CPU yields a finite average. Callers must keep every query row
// partially attended, which causal/image-prefix masks always do.)
int64_t top::FlexAttentionOp::getFLOPs() {
  int batch = getBatch();
  int M_q = getMq();
  int M_k = getMk();
  uint64_t qk_d = getQkD();
  uint64_t v_d = getVD();
  uint64_t q_head = getQHead();
  return batch * q_head * M_q * (M_k * qk_d + M_k * v_d) * 2;
}

LogicalResult top::FlexAttentionOp::init(InferenceParameter &p) {
  return success();
}
void top::FlexAttentionOp::deinit(InferenceParameter &p) {}

LogicalResult top::FlexAttentionOp::inference(InferenceParameter &p) {
  int batch = getBatch();
  int M_q = getMq();
  int M_k = getMk();
  uint64_t qk_d = getQkD();
  uint64_t v_d = getVD();
  uint64_t q_head = getQHead();
  auto kv_head = getKvHead();
  float scale = getScale().convertToDouble();
  int m_size = batch * q_head * M_q * M_k;
  bool has_mask = !module::isNone(getMask());
  int mask_size = getMaskSize();
  bool full_mask = (mask_size == 0);
  int q_pos_offset = M_k - M_q;
  auto qk_buffer = new float[m_size];
  // QK: q [B,mq,qh,qk_d] @ k^T -> [B,mq,qh,M_k] (contract over qk_d)
  dnnl_mm_gqa(p.inputs[0], p.inputs[1], qk_buffer, batch, q_head, kv_head, M_q,
              qk_d, M_k, 0);
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
  bool has_lse = getHasLse();
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
    // lse = max + log(sum). Loop i is [b, q_head, M_q]; write lse laid out as
    // [b, M_q, q_head] (transpose q_head/M_q), only when has_lse.
    if (has_lse) {
      int b_idx = i / (q_head * M_q);
      int rem = i % (q_head * M_q);
      int qh_idx = rem / M_q;
      int mq_idx = rem % M_q;
      int lse_offset = b_idx * (M_q * q_head) + mq_idx * q_head + qh_idx;
      p.outputs[1][lse_offset] = max + std::log(sum);
    }
  }
  // AV: w [B,mq,qh,M_k] @ v [B,mk,kvh,v_d] -> [B,mq,qh,v_d] (output dim v_d)
  float *temp = new float[batch * q_head * M_q * v_d];
  assert(temp != nullptr);
  dnnl_mm_gqa(qk_buffer, p.inputs[2], temp, batch, q_head, kv_head, M_q, M_k,
              v_d, 1);
  delete[] qk_buffer;
  tensor_hc_transpose(p.outputs[0], temp, batch, q_head, M_q, v_d);
  delete[] temp;

  return success();
}

// output: [b, M_q, q_head, v_d] (keep_dims) / [b, M_q, q_head*v_d].
// lse: [b, M_q, q_head, 1] (keep_dims) / [b, M_q, q_head] when has_lse; dummy
// [1] when has_lse is false (kernel does not write it). Cannot use
// common_shape_inference (asserts single output); set both result shapes.
void top::FlexAttentionOp::shape_inference() {
  auto out = getOutput();
  bool keep_dims = getKeepDims();
  bool has_lse = getHasLse();
  auto q_shape = module::getShape(getQueries()); // [B, mq, q_head, qk_d]
  assert(q_shape.size() == 4);
  if (keep_dims) {
    // output [B, mq, q_head, v_d]; lse [B, mq, q_head, 1] or dummy [1]
    module::setShapeOrVerify(
        out, {q_shape[0], q_shape[1], q_shape[2], (int64_t)getVD()});
    if (has_lse) {
      module::setShapeOrVerify(getLse(),
                               {q_shape[0], q_shape[1], q_shape[2], 1});
    } else {
      module::setShapeOrVerify(getLse(), {1});
    }
    return;
  }
  std::vector<int64_t> out_shape = {q_shape[0], q_shape[1],
                                    q_shape[2] * (int64_t)getVD()};
  module::setShapeOrVerify(out, out_shape);
  if (has_lse) {
    module::setShapeOrVerify(getLse(), {q_shape[0], q_shape[1], q_shape[2]});
  } else {
    module::setShapeOrVerify(getLse(), {1});
  }
}
