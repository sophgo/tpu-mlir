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

LogicalResult tpu::RecurrentGatedDeltaRuleOp::init(InferenceParameter &p) {
  return success();
}

void tpu::RecurrentGatedDeltaRuleOp::deinit(InferenceParameter &p) {}

LogicalResult tpu::RecurrentGatedDeltaRuleOp::inference(InferenceParameter &p) {
  // Input shapes:
  //   query:           [B, 1, num_k_heads, k_head_dim]
  //   key:             [B, 1, num_k_heads, k_head_dim]
  //   value:           [B, 1, num_v_heads, v_head_dim]
  //   g:               [B, 1, num_v_heads]
  //   beta:            [B, 1, num_v_heads]
  //   recurrent_state: [B, num_v_heads, k_head_dim, v_head_dim]
  // Output:
  //   attn_out:        [B, 1, num_v_heads, v_head_dim]

  auto v_shape = module::getShape(getValue());
  auto q_shape = module::getShape(getQuery());

  const int64_t B = v_shape[0];
  const int64_t num_v_heads = v_shape[2];
  const int64_t v_head_dim = v_shape[3];
  const int64_t num_k_heads = q_shape[2];
  const int64_t k_head_dim = q_shape[3];
  const int64_t num_group = num_v_heads / num_k_heads;
  const bool use_l2norm = getUseQkL2norm();
  const float scale = static_cast<float>(getScale().convertToDouble());

  float *in_query = p.inputs[0];
  float *in_key = p.inputs[1];
  float *in_value = p.inputs[2];
  float *in_g = p.inputs[3];
  float *in_beta = p.inputs[4];
  float *in_state = p.inputs[5];
  float *out_attn = p.outputs[0];

  // Copy query/key for optional L2 norm and scaling
  std::vector<float> query_buf(in_query,
                               in_query + B * num_k_heads * k_head_dim);
  std::vector<float> key_buf(in_key, in_key + B * num_k_heads * k_head_dim);

  if (use_l2norm) {
    auto l2norm_inplace = [](float *data, int64_t n, int64_t dim) {
      for (int64_t i = 0; i < n; i++) {
        float sum_sq = 0;
        for (int64_t d = 0; d < dim; d++)
          sum_sq += data[i * dim + d] * data[i * dim + d];
        float inv_norm = 1.0f / std::sqrt(sum_sq + 1e-6f);
        for (int64_t d = 0; d < dim; d++)
          data[i * dim + d] *= inv_norm;
      }
    };
    l2norm_inplace(query_buf.data(), B * num_k_heads, k_head_dim);
    l2norm_inplace(key_buf.data(), B * num_k_heads, k_head_dim);
  }

  // Scale query
  for (int64_t i = 0; i < B * num_k_heads * k_head_dim; i++)
    query_buf[i] *= scale;

  // Copy state for in-place modification
  int64_t state_total = B * num_v_heads * k_head_dim * v_head_dim;
  std::vector<float> state(in_state, in_state + state_total);

  for (int64_t b = 0; b < B; b++) {
    for (int64_t h = 0; h < num_v_heads; h++) {
      int64_t kh = h / num_group; // GQA: map v_head to k_head

      const float *q = query_buf.data() + (b * num_k_heads + kh) * k_head_dim;
      const float *k = key_buf.data() + (b * num_k_heads + kh) * k_head_dim;
      const float *v = in_value + (b * num_v_heads + h) * v_head_dim;
      float g_exp = std::exp(in_g[b * num_v_heads + h]);
      float beta_val = in_beta[b * num_v_heads + h];
      float *s = state.data() + (b * num_v_heads + h) * k_head_dim * v_head_dim;
      float *out = out_attn + (b * num_v_heads + h) * v_head_dim;

      // state = state * exp(g)
      for (int64_t i = 0; i < k_head_dim * v_head_dim; i++)
        s[i] *= g_exp;

      // kv_mem = state^T @ key  →  [v_head_dim]
      std::vector<float> kv_mem(v_head_dim, 0.0f);
      for (int64_t kk = 0; kk < k_head_dim; kk++) {
        for (int64_t j = 0; j < v_head_dim; j++) {
          kv_mem[j] += s[kk * v_head_dim + j] * k[kk];
        }
      }

      // delta = (value - kv_mem) * beta  →  [v_head_dim]
      std::vector<float> delta(v_head_dim);
      for (int64_t j = 0; j < v_head_dim; j++)
        delta[j] = (v[j] - kv_mem[j]) * beta_val;

      // state += key ⊗ delta  (outer product)
      for (int64_t kk = 0; kk < k_head_dim; kk++) {
        for (int64_t j = 0; j < v_head_dim; j++) {
          s[kk * v_head_dim + j] += k[kk] * delta[j];
        }
      }

      // output = state^T @ query  →  [v_head_dim]
      for (int64_t j = 0; j < v_head_dim; j++) {
        float sum = 0.0f;
        for (int64_t kk = 0; kk < k_head_dim; kk++) {
          sum += s[kk * v_head_dim + j] * q[kk];
        }
        out[j] = sum;
      }
    }
  }
  auto out_type = module::getStorageType(getAttnOut());
  auto num_elem = module::getNumElements(getAttnOut());
  if (out_type.isBF16()) {
    BF16(p.outputs[0], p.outputs[0], num_elem);
  } else if (out_type.isF16()) {
    F16(p.outputs[0], p.outputs[0], num_elem);
  }

  return success();
}

bool tpu::RecurrentGatedDeltaRuleOp::support_multi_core() { return true; }
