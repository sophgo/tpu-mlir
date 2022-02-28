
#pragma once

#include "oneapi/dnnl/dnnl.hpp"

namespace dnnl {
class MatMul {
public:
  MatMul();

  void setup(float *left, float *right, float *bias, float *output,
             int64_t batch, int64_t M, int64_t K, int64_t N, bool do_relu);

  void run();

private:
  engine eng;
  stream engine_stream;
  matmul primitive;
  std::unordered_map<int, memory> matmul_args;
};
} // namespace dnnl
