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
  // Derive shapes from the runtime operand shapes so that dynamic-shape
  // inference (where the actual K/V length is smaller than the compiled
  // `mk` attribute, e.g. history-kv with a partial cache) reads the right
  // number of rows instead of out-of-bounds memory.
  auto q_shape = module::getShape(getQueries());
  auto k_shape = module::getShape(getKeys());
  int batch = q_shape[0];
  int M_q = q_shape[1];
  uint64_t q_head = q_shape[2];
  uint64_t d = q_shape[3];
  int M_k = k_shape[1];
  auto kv_head = k_shape[2];
  float scale = getScale().convertToDouble();
  int64_t m_size = (int64_t)batch * q_head * M_q * M_k;
  bool has_mask = !module::isNone(getMask());
  int mask_size = getMaskSize();
  bool full_mask = (mask_size == 0);
  // Causal: query at row m attends to keys k <= m + q_pos_offset, where
  // q_pos_offset = M_k - M_q (prefix-cache + new prompt layout).
  int q_pos_offset = M_k - M_q;
  auto qk_buffer = new float[m_size];
  // Q * K
  dnnl_mm_gqa(p.inputs[0], p.inputs[1], qk_buffer, batch, q_head, kv_head, M_q,
              d, M_k, 0);
  // * scale
#pragma omp parallel for schedule(static, omp_schedule(m_size))
  for (int64_t i = 0; i < m_size; i++) {
    qk_buffer[i] *= scale;
    if (has_mask) {
      if (full_mask) {
        int64_t mask_offset = i % ((int64_t)M_q * M_k);
        qk_buffer[i] += p.inputs[3][mask_offset];
      } else {
        int mk = (int)(i % M_k);
        int mq = (int)((i / M_k) % M_q);
        if (mk > mq + q_pos_offset) {
          qk_buffer[i] += -1.0e9f;
        }
      }
    }
  }
  // do softmax
  int64_t outer_dim = (int64_t)batch * q_head * M_q;
#pragma omp parallel for schedule(static, omp_schedule(outer_dim))
  for (int64_t i = 0; i < outer_dim; i++) {
    int64_t offset = i * M_k;
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

// if keep_dims, output shape = input shape
// else input = [1, M_q, q_head, d], output = [1, M_q, q_head*d]
void top::FAttentionOp::shape_inference() {
  auto out = getOutput();
  bool keep_dims = getKeepDims();
  if (keep_dims) {
    common_shape_inference(getOperation());
    return;
  }
  auto in_shape = module::getShape(getQueries());
  assert(in_shape.size() == 4);
  std::vector<int64_t> out_shape;
  out_shape.push_back(in_shape[0]);
  out_shape.push_back(in_shape[1]);
  out_shape.push_back(in_shape[2] * in_shape[3]);
  module::setShapeOrVerify(out, out_shape);
}
