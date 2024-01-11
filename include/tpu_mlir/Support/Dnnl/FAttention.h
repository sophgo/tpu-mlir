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
class FAttention {
public:
  FAttention();

  void setup(float *queries, float *keys, float *values, float *mask,
             float *output, int64_t batch, int64_t M_q, int64_t M_k,
             int64_t head, int64_t d, float scale, int dtype = 0);
  void run();
  void deinit();

private:
  void *matmul0 = nullptr;
  void *binary = nullptr, *softmax = nullptr, *matmul1 = nullptr;

  std::shared_ptr<std::vector<float>> data_0, data_binary, data_softmax;
  float *p_mat0, *p_binary, *p_softmax, *p_mat1, *p_mat1_out, *p_mask, *p_output;
  int64_t num_elem, num_elem_out, dtype_, M_k_, M_q_, head_, batch_;
  float scale_;
};
} // namespace tpu_mlir
