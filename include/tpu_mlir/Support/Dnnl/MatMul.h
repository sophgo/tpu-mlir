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

  void right_init(float *right, int64_t right_zp, int64_t len);
  void setup(float *left, float *right, float *bias, float *output,
             int64_t batch, int64_t M, int64_t K, int64_t N, bool do_relu,
             double relu_limit, int64_t right_zp);

  void run();

private:
  engine eng;
  stream engine_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
  std::shared_ptr<std::vector<float>> bias0;
  float *p_right;
  std::shared_ptr<std::vector<float>> right_after_zp;
};
} // namespace tpu_mlir
