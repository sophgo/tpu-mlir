#pragma once
#include "oneapi/dnnl/dnnl.hpp"
using namespace dnnl;
namespace sophgo {
class MatMul {
public:
  MatMul();

  void setup(float *left, float *right, float *bias, float *output,
             int64_t batch, int64_t M, int64_t K, int64_t N, bool do_relu,
             int64_t rshift = 0, memory::data_type ldt = memory::data_type::f32,
             memory::data_type rdt = memory::data_type::f32,
             memory::data_type bdt = memory::data_type::f32,
             memory::data_type odt = memory::data_type::f32);

  void run();

private:
  engine eng;
  stream engine_stream;
  std::vector<primitive> net;
  std::vector<std::unordered_map<int, memory>> net_args;
};
} // namespace dnnl
