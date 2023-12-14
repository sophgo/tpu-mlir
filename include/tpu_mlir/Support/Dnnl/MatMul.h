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
class MatMul {
public:
  MatMul();

  void right_init(float *right, int64_t right_zp, int64_t batch,
                  int64_t batch_low, int64_t K, int64_t N,
                  bool right_transpose);
  void input_init(float *input, int64_t input_zp, int64_t batch,
                  int64_t batch_low, int64_t M, int64_t K,
                  bool input_transpose);
  void output_init(float *output, int64_t batch, int64_t batch_low, int64_t M,
                   int64_t N, bool output_transpose);
  // void setup(float *left, float *right, float *bias, float *output,
  //            int64_t batch, int64_t batch_low, int64_t M, int64_t K, int64_t N,
  //            bool do_relu, double relu_limit, int64_t right_zp,
  //            int64_t input_zp, bool right_transpose, bool input_transpose,
  //            bool output_transpose, bool hdim_is_batch);
  void setup(float *left, float *right, float *bias, float *output,
             int64_t batch, int64_t batch_low, int64_t M, int64_t K, int64_t N,
             bool do_relu, double relu_limit, int64_t right_zp,
             int64_t input_zp, bool right_transpose, bool input_transpose,
             bool output_transpose, bool hdim_is_batch, const std::vector<int64_t> &L_shape={},
             const std::vector<int64_t> &R_shape={}, int dims_merge_2_M=0);
  void run();

private:
  engine eng;
  stream engine_stream;
  primitive prim;
  dnnl::memory src_mem, weight_mem, bias_mem, dst_mem;
  std::shared_ptr<std::vector<float>> bias0;
  float *p_right, *p_input;
  float *origin_input, *origin_right, *origin_output;
  std::shared_ptr<std::vector<float>> right_after_init;
  std::shared_ptr<std::vector<float>> input_after_init;
  std::shared_ptr<std::vector<float>> output_after_trans;
  std::shared_ptr<std::vector<float>> input_broadcasted;
  std::shared_ptr<std::vector<float>> right_broadcasted;
  int64_t batch_, M_, N_, K_, right_zp_, input_zp_, l_b, r_b;
  std::vector<int64_t> L_shape_;
  std::vector<int64_t> R_shape_;
  bool need_broadcast_ = false;
  bool right_has_zp_ = 0, input_has_zp_ = 0;
  bool right_transpose_ = 0, input_transpose_ = 0, output_transpose_ = 0;
  bool hdim_is_batch_ = 0;
  int64_t batch_low_ = 1;
};
} // namespace tpu_mlir
