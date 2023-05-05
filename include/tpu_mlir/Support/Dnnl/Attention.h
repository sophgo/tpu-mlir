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
             float *out_weight, float *out_bias, float *musk, float *output,
             int64_t batch, int64_t M_q, int64_t M_k, int64_t N_q, int64_t N_k,
             int64_t d, float scale, bool add_result, int dtype=0);
  void run();
  void deinit();

private:
  void *matmulq = nullptr, *matmulk = nullptr, *matmulv = nullptr, *matmul0 = nullptr;
  void *binary = nullptr, *softmax = nullptr, *matmul1 = nullptr, *matmul_out = nullptr;

  std::shared_ptr<std::vector<float>> q_data, k_data, v_data, data_0, data_binary, data_softmax, data_1, data_out;
  float *p_queries, *p_keys, *p_values, *p_mat0, *p_binary, *p_softmax, *p_mat1, *p_mat1_out, *p_output;
  int64_t num_elem, num_elem_out, dtype_;
  float scale_;
  bool add_result_;
};
} // namespace tpu_mlir
