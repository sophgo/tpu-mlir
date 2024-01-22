//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace tpu_mlir {
class Attention {
public:
  Attention();

  void setup(float *input, float *keys, float *values,
             float *queries_weight, float *queries_bias, float *keys_weight,
             float *keys_bias, float *values_weight, float *values_bias,
             float *out_weight, float *out_bias, float *musk, float *table,
             float *output, int64_t *quant_param,
             int64_t batch, int64_t M_q, int64_t M_k, int64_t N_q, int64_t N_k,
             int64_t d, float scale, bool add_result, int dtype=0);
  void run();
  void deinit();

private:
  void *matmulq = nullptr, *matmulk = nullptr, *matmulv = nullptr, *matmul0 = nullptr;
  void *binary = nullptr, *softmax = nullptr, *matmul1 = nullptr, *matmul_out = nullptr;

  std::shared_ptr<std::vector<float>> q_data, k_data, v_data, data_0, data_binary, data_softmax, data_1, data_out;
  float *p_queries, *p_keys, *p_values, *p_mat0, *p_binary, *p_softmax, *p_mat1, *p_mat1_out, *p_output, *p_table, *p_musk;
  int64_t num_elem, num_elem_out, dtype_, M_k_, M_q_, batch_;
  int64_t q_mul, q_sft, q_zp, k_mul, k_sft, k_zp, v_mul, v_sft, v_zp, m0_mul, m0_sft, m0_zp, m1_mul, m1_sft, m1_zp, s_zp;
  float scale_;
  bool add_result_;
};

class ScaledDotProductAttention {
public:
  ScaledDotProductAttention();

  void setup(float *query, float *keys, float *values, float *masks, float *output,
             int64_t batch, int64_t head, int64_t query_len, int64_t seq_len, int64_t hidden_dim, int64_t value_dim,
             float scale, bool is_causal, int dtype=0);
  void run();
  void deinit();
private:
  float *query_, *keys_, *values_, *masks_, *output_;
  void *matmulqk, *softmax, *binary, *matmulv;
  float *qk, *qk_softmax, *qk_softmax_v, *p_binary;
  int64_t batch_, head_, seq_len_, hidden_dim_, value_dim_, query_len_;
  float scale_;
  bool is_causal_;
  int dtype_;
};

} // namespace tpu_mlir
